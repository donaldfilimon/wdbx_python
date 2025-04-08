# wdbx/ui/cli/cli.py
"""
Command-line interface for WDBX.

This module provides command-line tools for interacting with WDBX.
"""
import argparse
import importlib
import logging
import os
import sys
from typing import Any, Callable, Dict

import numpy as np

# Import core components directly from core module
from ...core import (
    DEFAULT_DATA_DIR,
    SHARD_COUNT,
    VECTOR_DIMENSION,
    WDBX,
    logger,
)

# Import the plugin loader
try:
    from .plugin_loader import (
        PluginRegistry,
        load_plugins,
        plugin_command,
        register_plugin_commands,
    )

    HAS_PLUGIN_SUPPORT = True
except ImportError:
    HAS_PLUGIN_SUPPORT = False
    logger.warning("Plugin support not available")

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
        "separator": "=",
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
        "separator": "-",
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
        "separator": "·",
    },
}

PLUGINS: Dict[str, PluginCommand] = {}


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WDBX - Wide Distributed Block Exchange")

    # Core settings
    core_group = parser.add_argument_group("Core Settings")
    core_group.add_argument(
        "--dimension",
        type=int,
        default=VECTOR_DIMENSION,
        help=f"Vector dimension (default: {VECTOR_DIMENSION})",
    )
    core_group.add_argument(
        "--shards", type=int, default=SHARD_COUNT, help=f"Number of shards (default: {SHARD_COUNT})"
    )
    core_group.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    core_group.add_argument(
        "--memory-limit",
        type=int,
        default=0,
        help="Memory limit in MB (0 = no limit)",
    )
    core_group.add_argument(
        "--storage-format",
        type=str,
        choices=["json", "binary", "hybrid"],
        default="json",
        help="Storage format for vector data (default: json)",
    )

    # Operating mode
    mode_group = parser.add_argument_group("Operation Mode")
    mode_group.add_argument("--server", action="store_true", help="Start HTTP server")
    mode_group.add_argument(
        "--host", type=str, default="127.0.0.1", help="HTTP server host (default: 127.0.0.1)"
    )
    mode_group.add_argument(
        "--port", type=int, default=5000, help="HTTP server port (default: 5000)"
    )
    mode_group.add_argument("--interactive", action="store_true", help="Start interactive mode")
    mode_group.add_argument(
        "--terminal-ui", action="store_true", help="Start the rich terminal UI dashboard"
    )
    mode_group.add_argument(
        "--simple-dashboard", action="store_true", help="Start simple monitoring dashboard"
    )
    mode_group.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    mode_group.add_argument(
        "--import", dest="import_file", type=str, help="Import vectors from a file"
    )
    mode_group.add_argument(
        "--export", dest="export_file", type=str, help="Export vectors to a file"
    )
    mode_group.add_argument("--query", type=str, help="Run a query and exit")

    # Logging and configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    config_group.add_argument("--config", type=str, help="Path to configuration file")
    config_group.add_argument(
        "--theme",
        type=str,
        default="default",
        choices=["default", "dark", "light", "green", "blue", "minimal", "terminal"],
        help="UI theme (default: default)",
    )
    config_group.add_argument("--debug", action="store_true", help="Enable debug mode")
    config_group.add_argument("--log-file", type=str, help="Path to log file")
    config_group.add_argument("--profile", action="store_true", help="Enable performance profiling")

    # Feature flags
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument("--enable-caching", action="store_true", help="Enable caching")
    feature_group.add_argument(
        "--cache-size", type=int, default=100, help="Number of items to cache (default: 100)"
    )
    feature_group.add_argument(
        "--enable-persona", action="store_true", help="Enable persona management"
    )
    feature_group.add_argument(
        "--content-filter",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="medium",
        help="Content filter level",
    )
    feature_group.add_argument(
        "--enable-compression", action="store_true", help="Enable data compression"
    )
    feature_group.add_argument(
        "--enable-encryption", action="store_true", help="Enable data encryption"
    )
    feature_group.add_argument(
        "--auto-backup", action="store_true", help="Enable automatic backups"
    )
    feature_group.add_argument(
        "--backup-interval", type=int, default=60, help="Backup interval in minutes (default: 60)"
    )

    # UI customization
    ui_group = parser.add_argument_group("UI Customization")
    ui_group.add_argument("--no-color", action="store_true", help="Disable colored output")
    ui_group.add_argument("--compact", action="store_true", help="Use compact display mode")
    ui_group.add_argument("--wide", action="store_true", help="Use wide display mode")
    ui_group.add_argument(
        "--refresh-rate", type=float, default=1.0, help="UI refresh rate in seconds (default: 1.0)"
    )
    ui_group.add_argument(
        "--show-memory-usage", action="store_true", help="Show memory usage in UI"
    )
    ui_group.add_argument("--show-cpu-usage", action="store_true", help="Show CPU usage in UI")
    ui_group.add_argument("--custom-logo", type=str, help="Path to custom logo file")

    # Plugins
    plugin_group = parser.add_argument_group("Plugins")
    plugin_group.add_argument(
        "--plugins", type=str, default="", help="Comma-separated list of plugins to load"
    )
    plugin_group.add_argument("--plugin-dir", type=str, help="Directory containing plugins")
    plugin_group.add_argument(
        "--list-plugins", action="store_true", help="List available plugins and exit"
    )
    plugin_group.add_argument("--plugin-config", type=str, help="Path to plugin configuration file")

    # API options
    api_group = parser.add_argument_group("API Options")
    api_group.add_argument("--api-key", type=str, help="API key for authentication")
    api_group.add_argument("--enable-cors", action="store_true", help="Enable CORS for API")
    api_group.add_argument(
        "--api-rate-limit", type=int, default=60, help="API rate limit per minute (default: 60)"
    )
    api_group.add_argument(
        "--api-timeout", type=int, default=30, help="API timeout in seconds (default: 30)"
    )
    api_group.add_argument("--api-docs", action="store_true", help="Generate API documentation")

    # Vector search options
    vector_group = parser.add_argument_group("Vector Search")
    vector_group.add_argument(
        "--distance-metric",
        type=str,
        choices=["cosine", "euclidean", "dot"],
        default="cosine",
        help="Distance metric for vector similarity (default: cosine)",
    )
    vector_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Threshold for vector similarity (default: 0.75)",
    )
    vector_group.add_argument(
        "--index-type",
        type=str,
        choices=["flat", "hnsw", "ivf"],
        default="flat",
        help="Vector index type (default: flat)",
    )

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


def run_interactive_mode(db: WDBX, config: Dict[str, Any]) -> None:
    """
    Run an interactive command-line interface for WDBX.

    Args:
        db: WDBX instance
        config: Configuration dictionary
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory

        has_prompt_toolkit = True
    except ImportError:
        logger.warning("prompt-toolkit not installed, using basic input")
        has_prompt_toolkit = False

    # Ensure history directory exists
    history_dir = os.path.expanduser("~/.wdbx")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, "cli_history")

    # Set up commands (basic ones + terminal UI commands)
    commands = get_basic_commands()

    # Load plugins if enabled
    plugin_registry = None
    if HAS_PLUGIN_SUPPORT and config.get("enable_plugins", False):
        logger.info("Loading plugins...")
        plugin_dirs = config.get("plugin_dirs", None)
        plugin_registry = load_plugins(plugin_dirs)
        register_plugin_commands(commands, plugin_registry)

        # Add plugin management command
        if plugin_registry and len(plugin_registry.loaded_plugins) > 0:
            commands["plugin"] = plugin_command(plugin_registry)
            logger.info(
                f"Loaded {len(plugin_registry.loaded_plugins)} plugins with "
                f"{sum(p.get('commands_registered', 0) for p in plugin_registry.loaded_plugins)} commands"
            )

    # Print welcome message
    show_welcome(config)

    # Start interactive loop
    if has_prompt_toolkit:
        # Set up prompt with history and auto-completion
        command_completer = WordCompleter(list(commands.keys()), ignore_case=True)
        session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=command_completer,
        )

        while True:
            try:
                # Get command with prompt-toolkit features
                cmd_input = session.prompt(get_prompt(config), complete_while_typing=True).strip()

                # Skip empty inputs
                if not cmd_input:
                    continue

                # Process the command
                if not process_command(cmd_input, commands, db, config):
                    break

            except KeyboardInterrupt:
                print("^C")
                continue
            except EOFError:
                print("^D")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                if config.get("debug", False):
                    import traceback

                    traceback.print_exc()
    else:
        # Basic input loop without prompt-toolkit features
        while True:
            try:
                # Get command with basic input
                cmd_input = input(get_prompt(config)).strip()

                # Skip empty inputs
                if not cmd_input:
                    continue

                # Process the command
                if not process_command(cmd_input, commands, db, config):
                    break

            except KeyboardInterrupt:
                print("^C")
                continue
            except EOFError:
                print("^D")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                if config.get("debug", False):
                    import traceback

                    traceback.print_exc()


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
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(description="WDBX Command Line Interface")

    # Add basic arguments
    parser.add_argument("--data-dir", help="Directory to store vector data")
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        help=f"Vector dimension (default: {DEFAULT_DIMENSION})",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=DEFAULT_NUM_SHARDS,
        help=f"Number of shards (default: {DEFAULT_NUM_SHARDS})",
    )
    parser.add_argument(
        "--memory-only", action="store_true", help="Run in memory-only mode (no persistence)"
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=DEFAULT_CACHE_TTL,
        help=f"Cache time-to-live in seconds (default: {DEFAULT_CACHE_TTL})",
    )

    # Command arguments
    parser.add_argument("--exec", help="Execute a command and exit")
    parser.add_argument("--monitor", action="store_true", help="Run the terminal UI monitor")
    parser.add_argument("--dashboard", action="store_true", help="Run the terminal dashboard")
    parser.add_argument(
        "--compact", action="store_true", help="Use compact display mode with --dashboard"
    )

    # UI customization
    parser.add_argument("--theme", default="default", help="UI theme (default, dark, light, ocean)")
    parser.add_argument("--no-colors", action="store_true", help="Disable colored output")
    parser.add_argument("--no-unicode", action="store_true", help="Disable unicode characters")

    # Web UI
    parser.add_argument("--web", action="store_true", help="Launch the web UI")
    parser.add_argument("--port", type=int, default=8501, help="Port for web UI (default: 8501)")

    # Plugin support
    parser.add_argument("--enable-plugins", action="store_true", help="Enable plugin support")
    parser.add_argument("--plugin-dirs", help="Comma-separated list of plugin directories")

    # Debug and logging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        set_log_level(logging.DEBUG)
    elif args.quiet:
        set_log_level(logging.WARNING)
    else:
        set_log_level(logging.INFO)

    # Create config from arguments
    config = {
        "data_dir": args.data_dir,
        "dimension": args.dimension,
        "num_shards": args.num_shards,
        "memory_only": args.memory_only,
        "cache_ttl": args.cache_ttl,
        "theme": args.theme,
        "use_colors": not args.no_colors,
        "use_unicode": not args.no_unicode,
        "debug": args.debug,
        "quiet": args.quiet,
        "port": args.port,
        "enable_plugins": args.enable_plugins,
    }

    # Handle plugin directories
    if args.plugin_dirs:
        config["plugin_dirs"] = args.plugin_dirs.split(",")

    # Initialize WDBX
    try:
        db = initialize_wdbx(config)
        logger.debug("WDBX initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize WDBX: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    # Run the appropriate mode
    if args.web:
        # Launch web UI
        try:
            from ..streamlit_app import run_streamlit_app

            run_streamlit_app(db, port=args.port)
        except ImportError as e:
            logger.error(f"Failed to launch web UI: {e}")
            logger.error("Make sure streamlit is installed (pip install streamlit)")
            return 1
    elif args.monitor:
        # Run terminal UI monitor
        from .terminal_ui import run_terminal_ui

        run_terminal_ui(db, config)
    elif args.dashboard:
        # Run dashboard
        from .terminal_ui import run_simple_dashboard

        run_simple_dashboard(db, config, compact=args.compact)
    elif args.exec:
        # Execute a command and exit
        commands = get_basic_commands()
        process_command(args.exec, commands, db, config)
    else:
        # Run interactive mode
        run_interactive_mode(db, config)

    return 0


if __name__ == "__main__":
    main()
