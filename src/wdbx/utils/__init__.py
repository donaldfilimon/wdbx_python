"""
Utility functions for the WDBX package.

This module contains various helper functions and utilities
that are used throughout the WDBX package.
"""

# Import utilities that should be exposed at the package level
# For now, this is empty as we'll need to migrate utility functions

# Make diagnostics available at the package level if present
try:
    # Create a context manager at the module level for timing operations
    from contextlib import contextmanager

    from .diagnostics import (
        get_monitor,
    )

    @contextmanager
    def time_this(operation_name: str):
        """
        Context manager for timing operations at the module level.

        Example:
            with wdbx.utils.time_this("my_operation"):
                # Do something that needs timing

        Args:
            operation_name: Name of the operation to time
        """
        monitor = get_monitor()
        with monitor.time_operation(operation_name):
            yield

except ImportError:
    pass
