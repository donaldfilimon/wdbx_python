"""
UI modules for WDBX.

This package contains various user interface components for WDBX,
including the Streamlit-based web UI for visualization and interaction.
"""

from pathlib import Path

# Expose UI-related constants
UI_DIR = Path(__file__).parent

__all__ = [
    "UI_DIR",
]
