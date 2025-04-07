"""
WDBX Debugger - Custom debugging tools for WDBX

This module provides extended debugging capabilities for the WDBX system,
including custom commands and enhanced visualization for debugging vector data.
"""

import pdb
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np


class WDBXDebugger(pdb.Pdb):
    """
    Custom debugger for WDBX with enhanced functionality.

    This debugger extends the standard Python debugger with WDBX-specific commands
    and visualization capabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the debugger with custom settings."""
        super().__init__(*args, **kwargs)
        self.prompt = "\033[1;36mwdbx-debug>\033[0m "
        self._vector_history: Dict[str, Any] = {}

    def do_meminfo(self, arg: str) -> None:
        """
        Display memory usage information for variables.

        Usage: meminfo [variable name(s)]
        If no variable names are provided, shows info for all local variables.
        """
        try:
            from pympler import asizeof
        except ImportError:
            self.message("Pympler module not available. Install with: pip install pympler")
            return

        frame = self.curframe
        local_vars = frame.f_locals

        if not arg:
            # Show memory usage for all variables
            var_names = sorted(local_vars.keys())
        else:
            # Show memory usage for specified variables
            var_names = arg.split()

        # Format as a table
        self.message("\033[1mVariable Memory Usage:\033[0m")
        self.message("\033[1m{:<20} {:<15} {:<15}\033[0m".format("Name", "Type", "Size"))
        self.message("-" * 50)

        total_size = 0
        for name in var_names:
            if name not in local_vars:
                self.message(f"Variable '{name}' not found in local scope")
                continue

            var = local_vars[name]
            size = asizeof.asizeof(var)
            total_size += size

            # Format size for display
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.2f} MB"

            self.message(f"{name:<20} {type(var).__name__:<15} {size_str:<15}")

        # Show total
        self.message("-" * 50)
        if total_size < 1024 * 1024:
            total_str = f"{total_size / 1024:.2f} KB"
        else:
            total_str = f"{total_size / (1024 * 1024):.2f} MB"
        self.message(f"\033[1mTotal: {total_str}\033[0m")

    do_mem = do_meminfo  # Alias

    def do_vec(self, arg: str) -> None:
        """
        Display information about a vector.

        Usage: vec <variable_name>
        """
        if not arg:
            self.message("Usage: vec <variable_name>")
            return

        frame = self.curframe
        local_vars = frame.f_locals

        var_name = arg.strip()
        if var_name not in local_vars:
            self.message(f"Variable '{var_name}' not found in local scope")
            return

        var = local_vars[var_name]

        # Check if it's a numpy array
        if isinstance(var, np.ndarray):
            self._show_vector_info(var_name, var)
        # Check if it's a list or similar that could be converted
        elif hasattr(var, "__len__") and hasattr(var, "__getitem__"):
            try:
                vec = np.array(var)
                self._show_vector_info(var_name, vec, was_converted=True)
            except BaseException:
                self.message(f"Could not convert '{var_name}' to a numpy array")
        else:
            self.message(f"'{var_name}' is not a vector or convertible to one")

    def _show_vector_info(self, name: str, vec: np.ndarray, was_converted: bool = False) -> None:
        """Show detailed information about a vector."""
        # Add to history
        self._vector_history[name] = vec

        # Display vector info
        conversion_note = " (converted to numpy array)" if was_converted else ""
        self.message(f"\033[1mVector '{name}'{conversion_note}:\033[0m")
        self.message(f"  Shape: {vec.shape}")
        self.message(f"  Dtype: {vec.dtype}")
        self.message(f"  Dimensions: {vec.ndim}")

        if vec.size > 0:
            self.message(f"  Min value: {vec.min()}")
            self.message(f"  Max value: {vec.max()}")
            self.message(f"  Mean: {vec.mean()}")
            self.message(f"  Norm: {np.linalg.norm(vec)}")

            # Show first few elements
            preview_size = min(5, vec.size)
            flat_vec = vec.flatten()
            preview = ", ".join(f"{x:.4f}" for x in flat_vec[:preview_size])

            if vec.size > preview_size:
                preview += ", ..."

            self.message(f"  Values: [{preview}]")

    def do_vecsim(self, arg: str) -> None:
        """
        Calculate similarity between two vectors.

        Usage: vecsim <vector1> <vector2>
        Uses the vectors from previous 'vec' commands or from local variables.
        """
        args = arg.strip().split()
        if len(args) != 2:
            self.message("Usage: vecsim <vector1> <vector2>")
            return

        vec1_name, vec2_name = args

        # Check history first
        vec1 = self._vector_history.get(vec1_name)
        vec2 = self._vector_history.get(vec2_name)

        # If not in history, check local variables
        frame = self.curframe
        local_vars = frame.f_locals

        if vec1 is None and vec1_name in local_vars:
            var = local_vars[vec1_name]
            if isinstance(var, np.ndarray):
                vec1 = var
            elif hasattr(var, "__len__") and hasattr(var, "__getitem__"):
                try:
                    vec1 = np.array(var)
                except BaseException:
                    pass

        if vec2 is None and vec2_name in local_vars:
            var = local_vars[vec2_name]
            if isinstance(var, np.ndarray):
                vec2 = var
            elif hasattr(var, "__len__") and hasattr(var, "__getitem__"):
                try:
                    vec2 = np.array(var)
                except BaseException:
                    pass

        # Check if we have both vectors
        if vec1 is None:
            self.message(f"Vector '{vec1_name}' not found")
            return
        if vec2 is None:
            self.message(f"Vector '{vec2_name}' not found")
            return

        # Make sure they're both flattened
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        # Check dimensions
        if vec1.size != vec2.size:
            self.message(f"Vector dimensions don't match: {vec1.size} vs {vec2.size}")
            return

        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0
        else:
            dot_product = np.dot(vec1, vec2)
            similarity = dot_product / (norm1 * norm2)

        # Calculate Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)

        # Display results
        self.message("\033[1mVector Similarity Analysis:\033[0m")
        self.message(f"  Vectors: '{vec1_name}' and '{vec2_name}'")
        self.message(f"  Cosine similarity: {similarity:.6f}")
        self.message(f"  Euclidean distance: {distance:.6f}")

    def do_wwatch(self, arg: str) -> None:
        """
        Set a WDBX watch expression to monitor variables over time.

        Usage: wwatch <expression>
        """
        if not arg:
            self.message("Usage: wwatch <expression>")
            return

        # Store in userspace
        if not hasattr(self, "_wdbx_watches"):
            self._wdbx_watches = []

        self._wdbx_watches.append(arg)
        self.message(f"Added watch for: {arg}")

    def do_wlist(self, arg: str) -> None:
        """
        List all current WDBX watches.

        Usage: wlist
        """
        if not hasattr(self, "_wdbx_watches") or not self._wdbx_watches:
            self.message("No watches set. Use 'wwatch <expression>' to add one.")
            return

        self.message("\033[1mWDBX Watches:\033[0m")
        for i, watch in enumerate(self._wdbx_watches, 1):
            self.message(f"  {i}. {watch}")

    def do_wclear(self, arg: str) -> None:
        """
        Clear WDBX watches.

        Usage: wclear [number]
        If number is specified, clears that watch; otherwise clears all.
        """
        if not hasattr(self, "_wdbx_watches") or not self._wdbx_watches:
            self.message("No watches set.")
            return

        if not arg:
            self._wdbx_watches = []
            self.message("All watches cleared.")
        else:
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(self._wdbx_watches):
                    watch = self._wdbx_watches.pop(idx)
                    self.message(f"Watch cleared: {watch}")
                else:
                    self.message("Invalid watch number. Use 'wlist' to see watches.")
            except ValueError:
                self.message("Invalid argument. Usage: wclear [number]")

    def trace_dispatch(self, frame, event, arg):
        """Override trace_dispatch to check watches."""
        if event == "line" and hasattr(self, "_wdbx_watches") and self._wdbx_watches:
            for watch in self._wdbx_watches:
                try:
                    value = eval(watch, frame.f_globals, frame.f_locals)
                    self.message(f"\033[1;33mWatch '{watch}' = {value}\033[0m")
                except BaseException:
                    pass

        return super().trace_dispatch(frame, event, arg)


def wdbx_debug(*, header: Optional[str] = None):
    """
    Start the WDBX debugger.
    This function can be used as a decorator or called directly.

    Args:
        header: Optional message to display when entering the debugger
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if header:
                print(f"\033[1;36m{header}\033[0m")

            debugger = WDBXDebugger()
            debugger.set_trace(sys._getframe().f_back)
            return func(*args, **kwargs)
        return wrapper

    # Check if being used as a decorator or direct call
    if callable(header):
        func = header
        header = None
        return decorator(func)
    return decorator


def debug_vector(vector: np.ndarray, name: str = "vector") -> None:
    """
    Print debug information about a vector.

    Args:
        vector: The vector to debug
        name: Name to display
    """
    print(f"\033[1;36mVector Debug: {name}\033[0m")
    print(f"  Shape: {vector.shape}")
    print(f"  Dtype: {vector.dtype}")
    print(f"  Min/Max: {vector.min():.4f} / {vector.max():.4f}")
    print(f"  Mean/Std: {vector.mean():.4f} / {vector.std():.4f}")
    print(f"  Norm: {np.linalg.norm(vector):.4f}")

    # Show first 5 elements
    if vector.size > 0:
        flat = vector.flatten()
        preview = ", ".join(f"{x:.4f}" for x in flat[:5])
        if vector.size > 5:
            preview += ", ..."
        print(f"  Values: [{preview}]")


def set_trace():
    """
    Start the WDBXDebugger at the caller's frame.
    Similar to pdb.set_trace() but using our custom debugger.
    """
    debugger = WDBXDebugger()
    debugger.set_trace(sys._getframe().f_back)


def post_mortem(traceback=None):
    """
    Start the WDBXDebugger for post-mortem debugging.

    Args:
        traceback: Traceback object to debug (if None, uses the last exception)
    """
    if traceback is None:
        traceback = sys.exc_info()[2]
    if traceback is None:
        print("No traceback available to debug.")
        return

    debugger = WDBXDebugger()
    debugger.reset()
    debugger.interaction(None, traceback)

# Setup simple exception hook


def _excepthook(exc_type, exc_value, exc_traceback):
    """Custom exception hook to offer post-mortem debugging."""
    # Print the regular traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    # Offer post-mortem debugging
    print("\033[1;36mWould you like to start the WDBX debugger for post-mortem analysis? (y/n)\033[0m")
    response = input().strip().lower()
    if response in ("y", "yes"):
        debugger = WDBXDebugger()
        debugger.reset()
        debugger.interaction(None, exc_traceback)


def enable_wdbx_debugger():
    """
    Enable the WDBX debugger for all uncaught exceptions.
    """
    sys.excepthook = _excepthook
    print("\033[1;32mWDBX debugger enabled. Post-mortem debugging will be offered for uncaught exceptions.\033[0m")


def disable_wdbx_debugger():
    """
    Disable the WDBX debugger and restore the default exception handling.
    """
    sys.excepthook = sys.__excepthook__
    print("\033[1;33mWDBX debugger disabled. Default exception handling restored.\033[0m")


class Debugger:
    """The WDBX Debugger provides tools for debugging, profiling, and tracing WDBX operations."""

    def __init__(self, target=None):
        self.logger = None  # Assuming a logger is set up

    def log_operation(self, operation_name, duration):
        """Log an operation's duration."""
        self.logger.debug("Operation logged")

    def log_message(self, message, level="INFO"):
        """Log a debug message."""
        self.logger.debug("Message logged")
