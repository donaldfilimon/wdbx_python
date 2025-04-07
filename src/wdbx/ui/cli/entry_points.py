"""
WDBX entry points for console scripts.

This module provides the entry points for the console scripts defined in setup.py.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def cli_main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the 'wdbx' command-line tool.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    try:
        # Import lazily to avoid circular imports
        import importlib
        cli_module = importlib.import_module("wdbx.ui.cli.cli")
        main_func = cli_module.main

        # Set sys.argv for argparse to use
        sys.argv = [sys.argv[0]] + (args if args else [])
        main_func()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def web_main(args: Optional[List[str]] = None) -> int:
    """
    Entry point for the 'wdbx-web' command-line tool.

    This function now launches the Streamlit-based UI.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    # Find the streamlit app
    module_dir = Path(__file__).parent.parent.parent
    streamlit_app_path = module_dir / "ui" / "streamlit_app.py"

    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at {streamlit_app_path}", file=sys.stderr)
        return 1

    # Try to import streamlit
    try:
        # Check if streamlit is available without importing it
        import importlib.util
        streamlit_available = importlib.util.find_spec("streamlit") is not None
        if not streamlit_available:
            raise ImportError("Streamlit not found")
    except ImportError:
        streamlit_available = False
        print("Streamlit not installed. Installing streamlit...", file=sys.stderr)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                  "streamlit", "plotly", "pandas"])
            print("Streamlit installed successfully.")
        except Exception as e:
            print(f"Failed to install Streamlit: {e}", file=sys.stderr)
            return 1

    # Launch the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app_path)]

    # Add any passed arguments
    if args:
        cmd.extend(args)

    print(f"Launching Streamlit UI: {streamlit_app_path}")

    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(f"Error launching Streamlit: {e}", file=sys.stderr)
        return 1
