#!/usr/bin/env python3
"""
WDBX Streamlit UI

This is the main entry point for the WDBX Streamlit UI.
"""

import logging

import streamlit as st  # type: ignore

from wdbx.ui.error_handling import display_error, ErrorDisplayer
from wdbx.ui.navigation import handle_navigation
from wdbx.ui.themes import apply_theme

# Configure logging
logger = logging.getLogger("wdbx.ui.app")


class StreamlitErrorDisplayer(ErrorDisplayer):
    """Streamlit implementation of the ErrorDisplayer protocol."""

    def error(self, message: str) -> None:
        st.error(message)


def main() -> None:
    """Main entry point for the Streamlit UI."""
    try:
        # Initialize UI
        st.set_page_config(page_title="WDBX UI", page_icon="🧊", layout="wide")

        # Apply theme
        apply_theme()

        # Create error displayer
        error_displayer = StreamlitErrorDisplayer()

        # Handle navigation
        handle_navigation()

    except Exception as e:
        display_error("UI Initialization Error", e, displayer=error_displayer, show_traceback=True)


if __name__ == "__main__":
    main()
