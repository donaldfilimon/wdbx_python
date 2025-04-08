"""
Command-line entry point for WDBX.
"""

import logging
import sys

# Only import what we actually use
from .ui.cli import main as cli_main

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for running WDBX from the command line."""
    # Redirect to the CLI main function
    sys.exit(cli_main())


if __name__ == "__main__":
    main()
