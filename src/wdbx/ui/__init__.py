"""
WDBX UI package.

This package provides UI components for interacting with WDBX,
including a Streamlit-based dashboard.
"""

import os
import sys
from pathlib import Path

# Expose UI-related constants
UI_DIR = Path(__file__).parent
ASSETS_DIR = UI_DIR / "assets"
DEFAULT_THEME = "dark"
SUPPORTED_THEMES = ["light", "dark", "blue", "green", "terminal"]
DEFAULT_UI_PORT = 8501

# Create assets directory if it doesn't exist
os.makedirs(ASSETS_DIR, exist_ok=True)

# Conditional imports for better error handling
try:
    from .cli.entry_points import cli_main, web_main

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

try:
    from .cli.terminal_ui import WDBXMonitor, run_terminal_ui

    TERMINAL_UI_AVAILABLE = True
except ImportError:
    TERMINAL_UI_AVAILABLE = False

# Expose the main UI components and functions
__all__ = [
    "UI_DIR",
    "ASSETS_DIR",
    "DEFAULT_THEME",
    "SUPPORTED_THEMES",
    "DEFAULT_UI_PORT",
]

# Add available components to __all__
if CLI_AVAILABLE:
    __all__.extend(["cli_main", "web_main"])

if TERMINAL_UI_AVAILABLE:
    __all__.extend(["run_terminal_ui", "WDBXMonitor"])

# Apply torch.classes fix for Streamlit
try:
    import torch

    # Fix "Examining the path of torch.classes raised" error when using Streamlit
    torch.classes.__path__ = []
    # Alternative fix that might work in some cases:
    # import os
    # torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except ImportError:
    # PyTorch not installed, no fix needed
    pass

__all__ = ["cli", "streamlit_app"]

# UI package path for resource location
UI_PACKAGE_DIR = Path(__file__).parent

# Version should match the parent package
try:
    from wdbx.version import __version__
except ImportError:
    __version__ = "0.1.0"
