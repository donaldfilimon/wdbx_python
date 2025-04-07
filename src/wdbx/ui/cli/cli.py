# wdbx/ui/cli/cli.py
"""
Command-line interface for WDBX.

This module provides command-line tools for interacting with WDBX.
"""
import argparse
import importlib
import logging
import sys
from typing import Any, Callable, Dict

import numpy as np

# Import core components directly from core module
from ...core import (
    DEFAULT_DATA_DIR,
    SHARD_COUNT,
    VECTOR_DIMENSION,
    WDBX,
    EmbeddingVector,
    WDBXConfig,
    create_wdbx,
    logger,
)

# Define a type alias for plugin commands for clarity
PluginCommand = Callable[[WDBX, str], None]

# Simple embedding function


def embed_text(text: str, dimension: int = 128) -> np.ndarray:
    """
    Create a simple embedding vector from text.

    Args:
        text: The text to embed
        dimension: The dimension of the embedding vector

    Returns:
        A numpy array containing the embedding vector
    """
    if not text:
        return np.random.rand(dimension).astype(np.float32)

    # Create a deterministic vector based on the text
    seed = sum(ord(c) for c in text)
    np.random.seed(seed)
    vec = np.random.rand(dimension).astype(np.float32)

    # Reset the random seed to avoid affecting other random operations
    np.random.seed(None)

    return vec


# CLI Configuration
CLI_THEMES = {
    "default": {
        "title": "\033[1;36m",
        "prompt": "\033[1;32m",
        "info": "\033[0;32m",
        "header": "\033[1;33m",
        "cmd": "\033[1m",
        "value": "\033[0;33m",
        "reset": "\033[0m",
        "error": "\033[1;31m",
        "success": "\033[1;32m",
        "separator": "="
    },
    "dark": {
        "title": "\033[1;35m",
        "prompt": "\033[1;34m",
        "info": "\033[0;34m",
        "header": "\033[1;37m",
        "cmd": "\033[1m",
        "value": "\033[0;36m",
        "reset": "\033[0m",
        "error": "\033[1;31m",
        "success": "\033[1;32m",
        "separator": "-"
    },
    "light": {
        "title": "\033[1;34m",
        "prompt": "\033[1;30m",
        "info": "\033[0;30m",
        "header": "\033[1;35m",
        "cmd": "\033[1m",
        "value": "\033[0;34m",
        "reset": "\033[0m",
        "error": "\033[1;31m",
        "success": "\033[1;32m",
        "separator": "·"
    }
}

PLUGINS: Dict[str, PluginCommand] = {}


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WDBX - Wide Distributed Block Exchange")

    # Core settings
    core_group = parser.add_argument_group("Core Settings")
    core_group.add_argument("--dimension", type=int, default=VECTOR_DIMENSION,
                            help=f"Vector dimension (default: {VECTOR_DIMENSION})")
    core_group.add_argument("--shards", type=int, default=SHARD_COUNT,
                            help=f"Number of shards (default: {SHARD_COUNT})")
    core_group.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                            help=f"Data directory (default: {DEFAULT_DATA_DIR})")

    # Operating mode
    mode_group = parser.add_argument_group("Operation Mode")
    mode_group.add_argument("--server", action="store_true",
                            help="Start HTTP server")
    mode_group.add_argument("--host", type=str, default="127.0.0.1",
                            help="HTTP server host (default: 127.0.0.1)")
    mode_group.add_argument("--port", type=int, default=5000,
                            help="HTTP server port (default: 5000)")
    mode_group.add_argument("--interactive", action="store_true",
                            help="Start interactive mode")

    # Logging and configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--log-level", type=str,
                              choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                              default="INFO", help="Set logging level")
    config_group.add_argument("--config", type=str, help="Path to configuration file")

    # Feature flags
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument("--enable-caching", action="store_true",
                               help="Enable caching")
    feature_group.add_argument("--cache-size", type=int, default=100,
                               help="Number of items to cache (default: 100)")
    feature_group.add_argument("--enable-persona", action="store_true",
                               help="Enable persona management")
    feature_group.add_argument("--content-filter", type=str,
                               choices=["none", "low", "medium", "high"],
                               default="medium", help="Content filter level")

    args = parser.parse_args()
    return vars(args)


def load_plugins(plugin_list: str) -> None:
    """
    Load plugins for the CLI

    Args:
        plugin_list: Comma-separated list of plugins to load
    """
    if not plugin_list:
        return

    plugins = plugin_list.split(",")
    for plugin_name in plugins:
        plugin_name = plugin_name.strip()
        if not plugin_name:
            continue

        try:
            logger.info(f"Loading plugin: {plugin_name}")

            # Try importing as a module
            module = importlib.import_module(f"wdbx_plugins.{plugin_name}")

            # Look for a register_commands function
            if hasattr(module, "register_commands"):
                module.register_commands(PLUGINS)
                logger.info(f"Successfully loaded plugin: {plugin_name}")
            else:
                logger.warning(f"Plugin {plugin_name} doesn't have a register_commands function")
        except ImportError as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")


def handle_plugin_command(db, cmd_name: str, args: str, plugins: Dict[str, Callable]) -> bool:
    """
    Handle a command from a plugin

    Args:
        db: WDBX instance
        cmd_name: Command name
        args: Command arguments
        plugins: Dictionary of plugin commands

    Returns:
        True if the command was handled, False otherwise
    """
    if cmd_name in plugins:
        try:
            # Call the plugin command handler
            plugins[cmd_name](db, args)
            return True
        except Exception as e:
            print(f"\033[1;31mError executing plugin command '{cmd_name}': {e}\033[0m")
            return True
    return False


def run_interactive_mode(db: WDBX, theme: str = "default", plugins: Dict = None) -> None:
    """
    Run interactive mode

    Args:
        db: WDBX instance
        theme: UI theme to use
        plugins: Dictionary of plugin commands
    """
    print(f"WDBX Interactive Mode ({theme} theme)")
    print("----------------------------")
    print(
        f"Using WDBX with {db.vector_dimension}-dimensional vectors across {db.num_shards} shards")
    print("\nAvailable commands:")
    print("  create <text> - Create an embedding and store it")
    print("  search <text> - Search for similar embeddings")
    print("  info - Show system information")
    print("  help - Show this help message")
    print("  exit - Exit interactive mode")

    # Add plugin commands to help if available
    if plugins:
        print("\nPlugin commands:")
        for cmd_name in sorted(plugins.keys()):
            cmd_func = plugins[cmd_name]
            doc = cmd_func.__doc__ or "No documentation available"
            print(f"  {cmd_name} - {doc.strip().split('.')[0]}")

    # Simple embedding function
    def embed_text(text: str) -> np.ndarray:
        # This is a very simple embedding function
        if not text:
            return np.random.rand(db.vector_dimension).astype(np.float32)
        # Create a deterministic vector based on the text
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        return np.random.rand(db.vector_dimension).astype(np.float32)

    while True:
        try:
            cmd = input("\nwdbx> ").strip()
            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "exit":
                print("Exiting interactive mode.")
                break

            if command == "help":
                # Re-display help
                print("Available commands:")
                print("  create <text> - Create an embedding and store it")
                print("  search <text> - Search for similar embeddings")
                print("  info - Show system information")
                print("  help - Show this help message")
                print("  exit - Exit interactive mode")
                if plugins:
                    print("\nPlugin commands:")
                    for cmd_name in sorted(plugins.keys()):
                        cmd_func = plugins[cmd_name]
                        doc = cmd_func.__doc__ or "No documentation available"
                        print(f"  {cmd_name} - {doc.strip().split('.')[0]}")

            elif command == "create":
                if not args:
                    print("Error: Text required for embedding creation")
                    continue

                # Create and store embedding
                vector = embed_text(args)
                embedding = EmbeddingVector(
                    vector=vector,
                    metadata={"text": args, "source": "interactive"}
                )
                vector_id = db.store_embedding(embedding)
                print(f"Created and stored embedding with ID: {vector_id}")

            elif command == "search":
                if not args:
                    print("Error: Text required for search")
                    continue

                # Search for similar vectors
                query_vector = embed_text(args)
                results = db.search_similar_vectors(query_vector, top_k=5)

                print(f"Search results for '{args}':")
                if not results:
                    print("  No results found")
                else:
                    for i, (vector_id, similarity) in enumerate(results):
                        print(
                            f"  Result {
                                i +
                                1}: Vector ID {vector_id}, Similarity: {
                                similarity:.4f}")

            elif command == "info":
                # Show system info
                stats = db.get_system_stats()
                print("System Information:")
                print(f"  Vector dimension: {db.vector_dimension}")
                print(f"  Number of shards: {db.num_shards}")
                print(f"  Data directory: {db.data_dir}")
                print(f"  Vectors stored: {stats['vectors']['stored']}")
                print(f"  Blocks created: {stats['blocks']['created']}")
                print(f"  Uptime: {stats['uptime_formatted']}")
                print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")

            elif plugins and command in plugins:
                # Execute plugin command
                try:
                    plugins[command](db, args)
                except Exception as e:
                    print(f"Error executing plugin command '{command}': {e}")

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for a list of available commands")

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_server(db: WDBX, host: str, port: int) -> None:
    """
    Run the HTTP server

    Args:
        db: WDBX instance
        host: Host address to bind to
        port: Port number to listen on
    """
    try:
        # Use the entry_points.web_main function instead of direct import
        # Store the WDBX instance in a global variable for the Streamlit app to access
        import builtins

        from .entry_points import web_main
        builtins._WDBX_INSTANCE = db

        logger.info(f"Starting Streamlit UI server on {host}:{port}")
        # Call web_main with appropriate arguments
        args = ["--server.address", host, "--server.port", str(port)]
        exit_code = web_main(args)

        if exit_code != 0:
            logger.error(f"Streamlit UI exited with code {exit_code}")
            sys.exit(exit_code)
    except ImportError as err:
        logger.error(f"Streamlit UI not available. Cannot start server: {err}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI"""
    # Parse command-line arguments
    args = parse_args()

    # Configure logging
    logging_level = getattr(logging, args.get("log_level", "INFO"))
    logging.getLogger().setLevel(logging_level)

    # Create configuration
    config = WDBXConfig(
        vector_dimension=args.get("dimension", VECTOR_DIMENSION),
        num_shards=args.get("shards", SHARD_COUNT),
        data_dir=args.get("data_dir", DEFAULT_DATA_DIR),
        enable_caching=args.get("enable_caching", False),
        cache_size=args.get("cache_size", 100),
        content_filter_level=args.get("content_filter", "medium")
    )

    # Create WDBX instance
    logger.info(
        f"Creating WDBX instance with {config.vector_dimension}-d vectors, {config.num_shards} shards")
    db = create_wdbx(
        vector_dimension=config.vector_dimension,
        num_shards=config.num_shards,
        data_dir=config.data_dir,
        enable_caching=config.enable_caching,
        cache_size=config.cache_size,
        content_filter_level=config.content_filter_level
    )

    try:
        # Determine operating mode
        if args.get("server", False):
            # Start HTTP server
            run_server(
                db=db,
                host=args.get("host", "127.0.0.1"),
                port=args.get("port", 5000)
            )
        elif args.get("interactive", False):
            # Start interactive mode
            run_interactive_mode(
                db=db,
                theme=args.get("theme", "default")
            )
        else:
            # No mode specified, default to interactive
            logger.info("No operation mode specified, starting interactive mode")
            run_interactive_mode(db)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Clean up
        logger.info("Shutting down WDBX")
        db.close()


if __name__ == "__main__":
    main()
