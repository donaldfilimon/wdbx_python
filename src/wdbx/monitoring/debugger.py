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

# Try to import ML backend components
try:
    from ..ml import JAX_AVAILABLE, TORCH_AVAILABLE, ArrayLike, get_ml_backend

    ML_BACKEND_AVAILABLE = True
    ml_backend = get_ml_backend()
except ImportError:
    ML_BACKEND_AVAILABLE = False
    JAX_AVAILABLE = False
    TORCH_AVAILABLE = False
    ml_backend = None


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

        # Setup ML backend if available
        self.ml_backend = ml_backend if ML_BACKEND_AVAILABLE else None
        if self.ml_backend:
            self.message(f"ML backend detected: {self.ml_backend.selected_backend}")

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
        # Check if it's a tensor from PyTorch or JAX
        elif TORCH_AVAILABLE and "torch.Tensor" in str(type(var)):
            # Convert PyTorch tensor to numpy
            try:
                vec = var.detach().cpu().numpy()
                self._show_vector_info(
                    var_name, vec, was_converted=True, original_type="PyTorch Tensor"
                )
            except Exception as e:
                self.message(f"Error converting PyTorch tensor: {e}")
        elif JAX_AVAILABLE and "jax" in str(type(var)):
            # Convert JAX array to numpy
            try:

                vec = np.array(var)
                self._show_vector_info(var_name, vec, was_converted=True, original_type="JAX Array")
            except Exception as e:
                self.message(f"Error converting JAX array: {e}")
        # Check if it's a list or similar that could be converted
        elif hasattr(var, "__len__") and hasattr(var, "__getitem__"):
            try:
                vec = np.array(var)
                self._show_vector_info(var_name, vec, was_converted=True)
            except Exception as e:
                self.message(f"Could not convert '{var_name}' to a numpy array: {e}")
        else:
            self.message(f"'{var_name}' is not a vector or convertible to one")

    def _show_vector_info(
        self, name: str, vec: np.ndarray, was_converted: bool = False, original_type: str = ""
    ) -> None:
        """Show detailed information about a vector."""
        # Add to history
        self._vector_history[name] = vec

        # Display vector info
        conversion_note = (
            f" (converted from {original_type})"
            if original_type
            else " (converted to numpy array)" if was_converted else ""
        )
        self.message(f"\033[1mVector '{name}'{conversion_note}:\033[0m")
        self.message(f"  Shape: {vec.shape}")
        self.message(f"  Dtype: {vec.dtype}")
        self.message(f"  Dimensions: {vec.ndim}")

        if vec.size > 0:
            self.message(f"  Min value: {vec.min()}")
            self.message(f"  Max value: {vec.max()}")
            self.message(f"  Mean: {vec.mean()}")
            self.message(f"  Norm: {np.linalg.norm(vec)}")

            # Add additional stats for large vectors
            if vec.size > 100:
                # Show percentiles
                p5 = np.percentile(vec, 5)
                p95 = np.percentile(vec, 95)
                self.message(f"  5th percentile: {p5:.4f}")
                self.message(f"  95th percentile: {p95:.4f}")

                # Show standard deviation
                std = np.std(vec)
                self.message(f"  Standard deviation: {std:.4f}")

                # Count zeros and near-zeros
                zero_count = np.count_nonzero(vec == 0)
                near_zero_count = np.count_nonzero(np.abs(vec) < 1e-6)
                self.message(f"  Zero count: {zero_count} ({zero_count/vec.size:.2%})")
                self.message(
                    f"  Near-zero count: {near_zero_count} ({near_zero_count/vec.size:.2%})"
                )

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
            elif TORCH_AVAILABLE and "torch.Tensor" in str(type(var)):
                vec1 = var.detach().cpu().numpy()
            elif JAX_AVAILABLE and "jax" in str(type(var)):
                vec1 = np.array(var)
            elif hasattr(var, "__len__") and hasattr(var, "__getitem__"):
                try:
                    vec1 = np.array(var)
                except Exception:
                    pass

        if vec2 is None and vec2_name in local_vars:
            var = local_vars[vec2_name]
            if isinstance(var, np.ndarray):
                vec2 = var
            elif TORCH_AVAILABLE and "torch.Tensor" in str(type(var)):
                vec2 = var.detach().cpu().numpy()
            elif JAX_AVAILABLE and "jax" in str(type(var)):
                vec2 = np.array(var)
            elif hasattr(var, "__len__") and hasattr(var, "__getitem__"):
                try:
                    vec2 = np.array(var)
                except Exception:
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

        # Check for size mismatch
        if vec1.size != vec2.size:
            self.message(
                f"Warning: Vector sizes don't match. {vec1_name}: {vec1.size}, {vec2_name}: {vec2.size}"
            )
            min_size = min(vec1.size, vec2.size)
            vec1 = vec1[:min_size]
            vec2 = vec2[:min_size]
            self.message(f"Using first {min_size} elements for comparison.")

        # Calculate cosine similarity
        try:
            if self.ml_backend:
                # Use our optimized ML backend
                similarity = self.ml_backend.cosine_similarity(vec1, vec2)
                self.message(
                    f"Using ML backend ({self.ml_backend.selected_backend}) for similarity calculation"
                )
            else:
                # Fallback to manual numpy calculation
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 == 0 or norm2 == 0:
                    similarity = 0
                else:
                    dot_product = np.dot(vec1, vec2)
                    similarity = dot_product / (norm1 * norm2)

                self.message("Using NumPy for similarity calculation")

            self.message(f"\033[1mCosine similarity: {similarity:.6f}\033[0m")

            # Also show Euclidean distance
            euclidean = np.linalg.norm(vec1 - vec2)
            self.message(f"Euclidean distance: {euclidean:.6f}")

            # Show magnitude comparison
            mag1 = np.linalg.norm(vec1)
            mag2 = np.linalg.norm(vec2)
            self.message(f"Magnitude {vec1_name}: {mag1:.6f}")
            self.message(f"Magnitude {vec2_name}: {mag2:.6f}")
            self.message(
                f"Magnitude ratio: {mag1/mag2:.6f}" if mag2 != 0 else "Magnitude ratio: inf"
            )

        except Exception as e:
            self.message(f"Error calculating similarity: {e}")

    def do_ml(self, arg: str) -> None:
        """
        Show ML backend information and debug ML tensors.

        Usage: ml [command]
        Commands:
            info - Show ML backend information
            convert <var> <backend> - Convert tensor to specified backend
            profile <op> - Profile a simple ML operation
        """
        if not ML_BACKEND_AVAILABLE:
            self.message("ML backend not available. Make sure the ML module is installed.")
            return

        args = arg.strip().split()
        if not args or args[0] == "info":
            # Show ML backend information
            self.message("\033[1mML Backend Information:\033[0m")
            self.message(f"Selected backend: {self.ml_backend.selected_backend}")
            self.message(f"JAX available: {JAX_AVAILABLE}")
            self.message(f"PyTorch available: {TORCH_AVAILABLE}")

            # Show more detailed backend info
            if self.ml_backend.selected_backend == "jax" and JAX_AVAILABLE:
                import jax

                self.message("\nJAX Devices:")
                for i, device in enumerate(jax.devices()):
                    self.message(f"  Device {i}: {device}")

            elif self.ml_backend.selected_backend == "pytorch" and TORCH_AVAILABLE:
                import torch

                self.message("\nPyTorch Information:")
                self.message(f"  Version: {torch.__version__}")
                self.message(f"  CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    self.message(f"  CUDA version: {torch.version.cuda}")
                    for i in range(torch.cuda.device_count()):
                        self.message(f"  Device {i}: {torch.cuda.get_device_name(i)}")

                has_mps = hasattr(torch, "mps") and torch.backends.mps.is_available()
                self.message(f"  MPS available: {has_mps}")

        elif args[0] == "convert" and len(args) >= 3:
            # Convert tensor between backends
            var_name = args[1]
            target_backend = args[2]

            if target_backend not in ["numpy", "jax", "torch"]:
                self.message(f"Unknown backend: {target_backend}")
                self.message("Supported backends: numpy, jax, torch")
                return

            # Check if the variable exists
            frame = self.curframe
            local_vars = frame.f_locals

            if var_name not in local_vars:
                self.message(f"Variable '{var_name}' not found in local scope")
                return

            var = local_vars[var_name]

            # Convert the variable
            try:
                if target_backend == "numpy":
                    result = self.ml_backend.to_numpy(var)
                    self.message(f"Converted '{var_name}' to NumPy array")
                elif target_backend == "jax":
                    if not JAX_AVAILABLE:
                        self.message("JAX is not available")
                        return
                    result = self.ml_backend.to_jax(var)
                    self.message(f"Converted '{var_name}' to JAX array")
                elif target_backend == "torch":
                    if not TORCH_AVAILABLE:
                        self.message("PyTorch is not available")
                        return
                    result = self.ml_backend.to_torch(var)
                    self.message(f"Converted '{var_name}' to PyTorch tensor")

                # Add to local variables
                converted_name = f"{var_name}_{target_backend}"
                frame.f_locals[converted_name] = result
                self.message(f"Result stored in variable: {converted_name}")

                # Show information about the result
                if target_backend == "numpy":
                    self._show_vector_info(converted_name, result)
                else:
                    # Convert back to numpy for display
                    np_result = self.ml_backend.to_numpy(result)
                    self._show_vector_info(
                        converted_name, np_result, was_converted=True, original_type=target_backend
                    )

            except Exception as e:
                self.message(f"Error converting variable: {e}")

        elif args[0] == "profile" and len(args) >= 2:
            # Profile a simple ML operation
            op_type = args[1]

            if op_type not in ["normalize", "cosine", "matmul"]:
                self.message(f"Unknown operation: {op_type}")
                self.message("Supported operations: normalize, cosine, matmul")
                return

            # Create test data
            try:
                import time

                # Generate random vectors
                size = 1000 if len(args) <= 2 else int(args[2])
                dim = 128 if len(args) <= 3 else int(args[3])

                self.message(f"Generating {size} random vectors of dimension {dim}...")
                data = np.random.randn(size, dim).astype(np.float32)

                # Profile the operation
                if op_type == "normalize":
                    self.message("\nProfiling normalize operation...")

                    # NumPy
                    start = time.time()
                    for i in range(min(100, size)):
                        vec = data[i]
                        norm = np.linalg.norm(vec)
                        _ = vec / norm if norm > 0 else vec
                    numpy_time = time.time() - start
                    self.message(f"  NumPy: {numpy_time:.6f} seconds")

                    # ML backend
                    start = time.time()
                    for i in range(min(100, size)):
                        _ = self.ml_backend.normalize(data[i])
                    backend_time = time.time() - start
                    self.message(
                        f"  {self.ml_backend.selected_backend.capitalize()}: {backend_time:.6f} seconds"
                    )

                    if numpy_time > 0:
                        speedup = numpy_time / backend_time
                        self.message(f"  Speedup: {speedup:.2f}x")

                elif op_type == "cosine":
                    self.message("\nProfiling cosine similarity operation...")

                    # NumPy
                    start = time.time()
                    for i in range(min(100, size - 1)):
                        vec1 = data[i]
                        vec2 = data[i + 1]
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        _ = np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
                    numpy_time = time.time() - start
                    self.message(f"  NumPy: {numpy_time:.6f} seconds")

                    # ML backend
                    start = time.time()
                    for i in range(min(100, size - 1)):
                        _ = self.ml_backend.cosine_similarity(data[i], data[i + 1])
                    backend_time = time.time() - start
                    self.message(
                        f"  {self.ml_backend.selected_backend.capitalize()}: {backend_time:.6f} seconds"
                    )

                    if numpy_time > 0:
                        speedup = numpy_time / backend_time
                        self.message(f"  Speedup: {speedup:.2f}x")

                elif op_type == "matmul":
                    self.message("\nProfiling matrix multiplication operation...")

                    # Use smaller matrices for this operation
                    mat_size = min(size, 500)
                    A = data[:mat_size, :dim]
                    B = data[:dim, :mat_size].T if dim < mat_size else data[:mat_size, :dim].T

                    # NumPy
                    start = time.time()
                    _ = np.matmul(A, B)
                    numpy_time = time.time() - start
                    self.message(f"  NumPy: {numpy_time:.6f} seconds")

                    # ML backend - backend specific implementation
                    if self.ml_backend.selected_backend == "jax" and JAX_AVAILABLE:
                        import jax
                        import jax.numpy as jnp

                        # Convert to JAX
                        A_jax = jnp.array(A)
                        B_jax = jnp.array(B)

                        # Compile with JIT
                        @jax.jit
                        def matmul(x, y):
                            return jnp.matmul(x, y)

                        # Warm-up JIT
                        _ = matmul(A_jax, B_jax)

                        # Benchmark
                        start = time.time()
                        _ = matmul(A_jax, B_jax)
                        backend_time = time.time() - start

                    elif self.ml_backend.selected_backend == "torch" and TORCH_AVAILABLE:
                        import torch

                        # Convert to PyTorch
                        A_torch = torch.tensor(A)
                        B_torch = torch.tensor(B)

                        # Check if CUDA is available
                        if torch.cuda.is_available():
                            A_torch = A_torch.cuda()
                            B_torch = B_torch.cuda()
                            self.message("  Using CUDA for PyTorch")

                        # Benchmark
                        start = time.time()
                        _ = torch.matmul(A_torch, B_torch)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()  # Wait for CUDA operations to finish
                        backend_time = time.time() - start

                    else:
                        # Fallback to numpy for other backends
                        backend_time = numpy_time

                    self.message(
                        f"  {self.ml_backend.selected_backend.capitalize()}: {backend_time:.6f} seconds"
                    )

                    if numpy_time > 0 and backend_time > 0:
                        speedup = numpy_time / backend_time
                        self.message(f"  Speedup: {speedup:.2f}x")

            except Exception as e:
                self.message(f"Error profiling operation: {e}")
                traceback.print_exc()

        else:
            self.message("Usage: ml [command]")
            self.message("Commands:")
            self.message("  info - Show ML backend information")
            self.message("  convert <var> <backend> - Convert tensor to specified backend")
            self.message("  profile <op> [size] [dim] - Profile a simple ML operation")
            self.message("    Supported operations: normalize, cosine, matmul")

    def do_wwatch(self, arg: str) -> None:
        """
        Watch a variable's value and print when it changes.

        Usage: wwatch <variable_name>
        """
        if not arg:
            self.message("Usage: wwatch <variable_name>")
            return

        frame = self.curframe
        local_vars = frame.f_locals

        var_name = arg.strip()
        if var_name not in local_vars:
            self.message(f"Variable '{var_name}' not found in local scope")
            return

        self.message(f"Watching variable: {var_name}")
        return

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
    print(
        "\033[1;36mWould you like to start the WDBX debugger for post-mortem analysis? (y/n)\033[0m"
    )
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
    print(
        "\033[1;32mWDBX debugger enabled. Post-mortem debugging will be offered for uncaught exceptions.\033[0m"
    )


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
