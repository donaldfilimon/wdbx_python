"""
Command Line Interface (CLI) for WDBX.

Provides commands to interact with the WDBX system, including
server management, data inspection, configuration, and visualization.
"""

from .entry_points import (
    benchmark_main,
    cli_main,
    dashboard_main,
    parse_custom_args,
    print_banner,
    web_main,
)
from .cli import (
    CLI_THEMES,
    embed_text,
    handle_plugin_command,
    main,
    parse_args,
    run_interactive_mode,
    run_server,
)
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wdbx.cli")

# Import CLI module functions

# Import entry points

# Import plugin loader components
try:
    from .plugin_loader import (
        PluginRegistry,
        load_plugins,
        plugin_command,
        register_plugin_commands,
    )

    HAS_PLUGIN_SUPPORT = True
except ImportError as e:
    logger.debug(f"Plugin loader not available: {e}")
    HAS_PLUGIN_SUPPORT = False

# Define plugin command type
PluginCommand = Callable[..., Any]

# Plugin registry
PLUGIN_REGISTRY: Dict[str, PluginCommand] = {}

# Import terminal UI components conditionally to avoid import errors
try:
    from .terminal_ui import (
        WDBXMonitor,
        create_dashboard,
        run_simple_dashboard,
        run_terminal_ui,
    )

    terminal_ui_available = True
except ImportError as e:
    logger.debug(f"Terminal UI not available: {e}")
    terminal_ui_available = False

# Expose all public components
__all__ = [
    # Entry points
    "cli_main",
    "web_main",
    "dashboard_main",
    "benchmark_main",
    "parse_custom_args",
    "print_banner",
    # CLI components
    "main",
    "parse_args",
    "run_interactive_mode",
    "run_server",
    "CLI_THEMES",
    "handle_plugin_command",
    "embed_text",
    "PLUGIN_REGISTRY",
]

# Add plugin loader components if available
if HAS_PLUGIN_SUPPORT:
    __all__ += [
        "load_plugins",
        "PluginRegistry",
        "register_plugin_commands",
        "plugin_command",
        "HAS_PLUGIN_SUPPORT",
    ]
else:
    __all__ += ["HAS_PLUGIN_SUPPORT"]

# Add terminal UI components if available
if terminal_ui_available:
    __all__ += [
        "run_terminal_ui",
        "run_simple_dashboard",
        "WDBXMonitor",
        "create_dashboard",
        "terminal_ui_available",
    ]
else:
    __all__ += ["terminal_ui_available"]


# Helper function to register a plugin command
def register_plugin(name: str, command_func: PluginCommand) -> None:
    """
    Register a plugin command with the CLI.

    Args:
        name: Name of the command
        command_func: Function to execute for the command
    """
    PLUGIN_REGISTRY[name] = command_func
    logger.debug(f"Registered plugin command: {name}")


# Helper function to get a formatted list of available commands
def get_available_commands() -> str:
    """
    Get a formatted list of available commands.

    Returns:
        Formatted string with available commands
    """
    standard_commands = [
        ("create", "Create an embedding and store it"),
        ("search", "Search for similar embeddings"),
        ("info", "Show system information"),
        ("help", "Show help message"),
        ("exit", "Exit interactive mode"),
    ]

    plugin_commands = []
    for cmd_name, cmd_func in PLUGIN_REGISTRY.items():
        doc = cmd_func.__doc__ or "No documentation available"
        description = doc.strip().split(".")[0]
        plugin_commands.append((cmd_name, description))

    # Format the commands
    result = "\n\033[1;36mStandard Commands:\033[0m\n"
    for cmd, desc in standard_commands:
        result += f"  \033[1;32m{cmd}\033[0m - {desc}\n"

    if plugin_commands:
        result += "\n\033[1;36mPlugin Commands:\033[0m\n"
        for cmd, desc in sorted(plugin_commands):
            result += f"  \033[1;33m{cmd}\033[0m - {desc}\n"

    return result
