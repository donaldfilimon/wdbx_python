"""
Command Line Interface (CLI) for WDBX.

Provides commands to interact with the WDBX system, including
server management, data inspection, and configuration.
"""

from .entry_points import cli_main, web_main

# Import these conditionally to avoid circular imports


def _import_cli():
    from wdbx.ui.cli.cli import main, parse_args, run_interactive_mode, run_server
    from wdbx.ui.cli.terminal_ui import TerminalUI
    return main, parse_args, run_interactive_mode, run_server, TerminalUI


__all__ = [
    "cli_main",
    "web_main",
    # Add other exposed CLI functions if needed
    # from .cli import main, parse_args, run_interactive_mode, run_server
    # from .terminal_ui import TerminalUI
]
