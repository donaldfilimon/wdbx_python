"""
Test helpers for WDBX.

This module provides utilities for testing WDBX components, including test case mixins,
fixtures, and helper functions. It has enhanced support for:
- Windows compatibility
- Resource management
- Performance testing
- Memory usage tracking
- Mocking and fixtures
- Async testing
"""

import asyncio
import gc
import io
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast
from unittest import mock

import numpy as np

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    # Import WDBX modules, with fallbacks for import errors
    from ..config.config_manager import get_config_manager
except ImportError:
    # Mock for when running tests outside of package
    def get_config_manager():
        return type(
            "MockConfigManager",
            (),
            {"load_all": lambda: None, "_config_values": {}, "_loaded_sources": set()},
        )()


try:
    from ..utils.logging_utils import get_logger, log_context
except ImportError:
    # Mock logging utilities
    def get_logger(name):
        return logging.getLogger(name)

    @contextmanager
    def log_context(**kwargs):
        yield


try:
    from ..utils.memory_utils import get_memory_usage
except ImportError:
    # Fallback memory usage implementation
    def get_memory_usage():
        try:
            import psutil

            process = psutil.Process()
            rss = process.memory_info().rss
            percent = process.memory_percent()
            return {
                "process": {"rss": rss, "percent": percent},
                "system": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                },
            }
        except ImportError:
            # Basic implementation when psutil isn't available
            return {
                "process": {"rss": 0, "percent": 0},
                "system": {"total": 0, "available": 0, "percent": 0},
            }


# Initialize logger
logger = get_logger("WDBX.Testing")

# Type variables for better type hinting
T = TypeVar("T")
CallableT = TypeVar("CallableT", bound=Callable[..., Any])


class WDBXTestCase(unittest.TestCase):
    """
    Base test case for WDBX tests.

    Provides common functionality for WDBX tests, including:
    - Temporary directory creation and cleanup
    - Configuration management
    - Log capture
    - Performance measurement
    - Cross-platform compatibility
    """

    # Class-level attributes
    temp_dir: str = ""
    is_windows: bool = platform.system() == "Windows"
    original_env: Dict[str, str] = {}

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level resources."""
        super().setUpClass()

        # Save original environment
        cls.original_env = dict(os.environ)

        # Create a temporary directory for tests
        cls.temp_dir = tempfile.mkdtemp(prefix="wdbx_test_")

        # Use Path for cross-platform compatibility
        temp_path = Path(cls.temp_dir)
        data_path = temp_path / "data"
        logs_path = temp_path / "logs"
        config_path = temp_path / "config"

        # Configure the test environment
        os.environ["WDBX_ENV"] = "testing"
        os.environ["WDBX_DATA_DIR"] = str(data_path)
        os.environ["WDBX_LOG_DIR"] = str(logs_path)
        os.environ["WDBX_CONFIG_DIR"] = str(config_path)
        os.environ["WDBX_LOG_TO_FILE"] = "false"
        os.environ["WDBX_MEMORY_OPTIMIZATION_ENABLED"] = "false"

        # Create standard directories
        data_path.mkdir(exist_ok=True, parents=True)
        logs_path.mkdir(exist_ok=True, parents=True)
        config_path.mkdir(exist_ok=True, parents=True)

        # Reset config manager to pick up new environment
        try:
            config_manager = get_config_manager()
            config_manager._config_values = {}
            config_manager._loaded_sources = set()
            config_manager.load_all()
        except Exception as e:
            logger.warning(f"Failed to initialize config manager: {e}")

        logger.info(f"Test class {cls.__name__} set up, temp dir: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up class-level resources."""
        # Remove temporary directory with error handling
        if hasattr(cls, "temp_dir") and cls.temp_dir:
            try:
                temp_path = Path(cls.temp_dir)
                if temp_path.exists():
                    # On Windows, sometimes file handles aren't released immediately
                    if cls.is_windows:
                        attempts = 3
                        for i in range(attempts):
                            try:
                                shutil.rmtree(cls.temp_dir)
                                break
                            except PermissionError:
                                if i < attempts - 1:
                                    logger.warning(
                                        "Failed to remove temp dir, retrying after GC..."
                                    )
                                    gc.collect()
                                    time.sleep(0.5)
                                else:
                                    logger.warning(
                                        f"Could not remove temporary directory {cls.temp_dir}"
                                    )
                    else:
                        shutil.rmtree(cls.temp_dir)

                    logger.info(f"Removed temporary directory {cls.temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {e}")

        # Restore environment variables
        try:
            os.environ.clear()
            os.environ.update(cls.original_env)
        except Exception as e:
            logger.warning(f"Error restoring environment variables: {e}")

        # Reset config manager
        try:
            config_manager = get_config_manager()
            config_manager._config_values = {}
            config_manager._loaded_sources = set()
        except Exception:
            pass

        # Run garbage collection to clean up resources
        gc.collect()

        super().tearDownClass()

    def setUp(self) -> None:
        """Set up test-level resources."""
        super().setUp()

        # Create a test-specific temporary directory with Path for cross-platform compatibility
        test_name = self.id().split(".")[-1]
        self.test_temp_dir = Path(self.temp_dir) / test_name
        self.test_temp_dir.mkdir(exist_ok=True, parents=True)

        # Set up log context for the test
        try:
            self._log_context = log_context(
                component="Test", tags=["test", self.__class__.__name__], test_name=self.id()
            )
            self._log_context.__enter__()
        except Exception as e:
            logger.warning(f"Failed to set up log context: {e}")

            # Create a dummy context manager to prevent errors in tearDown
            @contextmanager
            def dummy_context():
                yield

            self._log_context = dummy_context()
            self._log_context.__enter__()

        # Save start time for performance tracking
        self._start_time = time.time()
        logger.info(f"Starting test {self.id()}")

    def tearDown(self) -> None:
        """Clean up test-level resources."""
        # Calculate test duration for performance tracking
        duration = time.time() - self._start_time
        logger.info(f"Finished test {self.id()} in {duration:.6f} seconds")

        # Exit log context with proper exception handling
        try:
            self._log_context.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error exiting log context: {e}")

        # Clean up test-specific temp directory with error handling
        if hasattr(self, "test_temp_dir"):
            try:
                temp_dir = Path(self.test_temp_dir)
                if temp_dir.exists():
                    # On Windows, sometimes files can't be deleted immediately
                    if self.is_windows:
                        try:
                            shutil.rmtree(temp_dir)
                        except (PermissionError, OSError):
                            logger.warning(f"Could not remove test temp directory {temp_dir}")
                    else:
                        shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up test temporary directory: {e}")

        # Run garbage collection to release resources
        gc.collect()

        super().tearDown()

    @contextmanager
    def capture_logs(self, level: str = "INFO") -> Generator[io.StringIO, None, None]:
        """
        Capture logs during a test.

        Args:
            level: Minimum log level to capture

        Yields:
            StringIO object containing captured logs
        """
        log_capture = io.StringIO()

        # Create a handler that writes to the StringIO object
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)

        # Add the handler to the root logger
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(min(original_level, getattr(logging, level.upper())))
        root_logger.addHandler(handler)

        try:
            yield log_capture
        finally:
            # Remove the handler and restore original level
            root_logger.removeHandler(handler)
            root_logger.setLevel(original_level)

    @contextmanager
    def assert_logs(
        self, level: str = "INFO", message: Optional[str] = None, count: Optional[int] = None
    ) -> Generator[io.StringIO, None, None]:
        """
        Assert that logs are generated during a block.

        Args:
            level: Minimum log level to capture
            message: Optional message that should appear in the logs
            count: Optional number of times the message should appear

        Yields:
            StringIO object containing captured logs
        """
        with self.capture_logs(level) as logs:
            yield logs

            log_content = logs.getvalue()
            self.assertTrue(log_content, f"No logs were generated at level {level} or higher")

            if message:
                if count is not None:
                    actual_count = log_content.count(message)
                    self.assertEqual(
                        actual_count,
                        count,
                        f"Expected message to appear {count} times, but found {actual_count} times: {message}",
                    )
                else:
                    self.assertIn(
                        message, log_content, f"Expected message not found in logs: {message}"
                    )

    @contextmanager
    def measure_time(
        self, threshold: Optional[float] = None
    ) -> Generator[Dict[str, float], None, None]:
        """
        Measure execution time of a block of code.

        Args:
            threshold: Optional threshold in seconds for test to pass

        Yields:
            Dictionary with timing information (start, end, duration)
        """
        result = {"start": time.time(), "end": 0, "duration": 0}

        try:
            yield result
        finally:
            result["end"] = time.time()
            result["duration"] = result["end"] - result["start"]

            logger.info(f"Execution time: {result['duration']:.6f} seconds")

            if threshold is not None:
                self.assertLessEqual(
                    result["duration"],
                    threshold,
                    f"Execution took {result['duration']:.6f}s, exceeding threshold of {threshold}s",
                )

    @contextmanager
    def retry_assertion(
        self, max_attempts: int = 3, delay: float = 0.5
    ) -> Generator[None, None, None]:
        """
        Retry assertions within a block until they succeed or max attempts are reached.

        Useful for testing async conditions or UI that may take time to update.

        Args:
            max_attempts: Maximum number of attempts
            delay: Delay in seconds between attempts

        Yields:
            None
        """
        for attempt in range(max_attempts):
            try:
                yield
                # If we get here, the assertions passed
                return
            except AssertionError:
                if attempt == max_attempts - 1:
                    # This was the last attempt, so re-raise
                    raise
                time.sleep(delay)

    @contextmanager
    def captured_output(self) -> Generator[Tuple[io.StringIO, io.StringIO], None, None]:
        """
        Capture stdout and stderr output.

        Yields:
            Tuple of (stdout_capture, stderr_capture)
        """
        new_out, new_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield new_out, new_err
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def create_temp_file(
        self, content: str, name: Optional[str] = None, suffix: str = ".txt"
    ) -> Path:
        """
        Create a temporary file with the given content.

        Args:
            content: Content to write to the file
            name: Optional name for the file (defaults to a random name)
            suffix: File extension to use (default: .txt)

        Returns:
            Path to the created file
        """
        if not name:
            name = f"temp_{int(time.time())}_{hash(content) % 10000:04d}{suffix}"

        file_path = Path(self.test_temp_dir) / name

        # Ensure consistent line endings for cross-platform compatibility
        normalized_content = content.replace("\r\n", "\n")

        # Use utf-8 encoding for all files
        file_path.write_text(normalized_content, encoding="utf-8")

        logger.debug(f"Created temporary file: {file_path}")
        return file_path

    def create_temp_dir(self, name: Optional[str] = None) -> Path:
        """
        Create a temporary subdirectory.

        Args:
            name: Optional name for the directory (defaults to a random name)

        Returns:
            Path to the created directory
        """
        if not name:
            name = f"dir_{int(time.time())}_{id(name) % 10000:04d}"

        dir_path = Path(self.test_temp_dir) / name
        dir_path.mkdir(exist_ok=True, parents=True)

        logger.debug(f"Created temporary directory: {dir_path}")
        return dir_path

    def assert_files_equal(
        self,
        file1: Union[str, Path],
        file2: Union[str, Path],
        ignore_line_endings: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """
        Assert that two files have identical content.

        Args:
            file1: Path to first file
            file2: Path to second file
            ignore_line_endings: Whether to normalize line endings before comparison
            encoding: File encoding to use
        """
        path1 = Path(file1)
        path2 = Path(file2)

        self.assertTrue(path1.exists(), f"File does not exist: {path1}")
        self.assertTrue(path2.exists(), f"File does not exist: {path2}")

        content1 = path1.read_text(encoding=encoding)
        content2 = path2.read_text(encoding=encoding)

        if ignore_line_endings:
            content1 = content1.replace("\r\n", "\n")
            content2 = content2.replace("\r\n", "\n")

        self.assertEqual(content1, content2, f"Files are not equal: {path1} vs {path2}")

    def assert_json_equal(
        self, json1: Any, json2: Any, msg: Optional[str] = None, ignore_order: bool = False
    ) -> None:
        """
        Assert that two JSON objects are equal.

        Args:
            json1: First JSON object
            json2: Second JSON object
            msg: Optional message to display
            ignore_order: Whether to ignore the order of list items
        """

        def format_json(obj: Any) -> str:
            """Format a JSON object for display."""
            return json.dumps(obj, sort_keys=True, indent=2)

        if ignore_order and isinstance(json1, list) and isinstance(json2, list):
            # Sort lists by their JSON representation for order-independent comparison
            if len(json1) != len(json2):
                self.assertEqual(
                    len(json1),
                    len(json2),
                    msg or f"JSON lists have different lengths: {len(json1)} vs {len(json2)}",
                )

            # Try to sort if items are comparable
            try:
                sorted1 = sorted(json1, key=lambda x: json.dumps(x, sort_keys=True))
                sorted2 = sorted(json2, key=lambda x: json.dumps(x, sort_keys=True))
                json1, json2 = sorted1, sorted2
            except (TypeError, ValueError):
                # If sorting fails, do element-by-element comparison
                for item1 in json1:
                    found = False
                    for item2 in json2:
                        try:
                            self.assert_json_equal(item1, item2, ignore_order=True)
                            found = True
                            break
                        except AssertionError:
                            continue
                    self.assertTrue(found, f"Item not found in second list: {format_json(item1)}")
                return

        formatted1 = format_json(json1)
        formatted2 = format_json(json2)

        self.assertEqual(
            formatted1,
            formatted2,
            msg or f"JSON objects are not equal:\n{formatted1}\nvs\n{formatted2}",
        )

    def assert_dict_contains_subset(
        self, subset: Dict[Any, Any], dictionary: Dict[Any, Any], msg: Optional[str] = None
    ) -> None:
        """
        Assert that a dictionary contains all key-value pairs from a subset.

        Args:
            subset: Expected subset of key-value pairs
            dictionary: Dictionary to check
            msg: Optional message to display
        """
        missing_keys = []
        mismatched_values = {}

        for key, expected_value in subset.items():
            if key not in dictionary:
                missing_keys.append(key)
            elif dictionary[key] != expected_value:
                mismatched_values[key] = (expected_value, dictionary[key])

        if missing_keys or mismatched_values:
            formatted_subset = json.dumps(subset, sort_keys=True, indent=2)
            formatted_dict = json.dumps(dictionary, sort_keys=True, indent=2)
            error_msg = f"Dictionary does not contain expected subset:\n{formatted_subset}\nActual dictionary:\n{formatted_dict}"

            if missing_keys:
                error_msg += f"\nMissing keys: {missing_keys}"

            if mismatched_values:
                error_msg += "\nMismatched values:"
                for key, (expected, actual) in mismatched_values.items():
                    error_msg += f"\n  {key}: expected {expected}, got {actual}"

            self.fail(msg or error_msg)

    def assert_arrays_equal(
        self,
        array1: np.ndarray,
        array2: np.ndarray,
        decimal: int = 7,
        msg: Optional[str] = None,
        element_wise: bool = False,
    ) -> None:
        """
        Assert that two numpy arrays are equal.

        Args:
            array1: First array
            array2: Second array
            decimal: Precision for floating point comparison
            msg: Optional message to display
            element_wise: If True, print detailed element-wise differences on failure
        """
        try:
            np.testing.assert_almost_equal(array1, array2, decimal=decimal)
        except AssertionError as e:
            if not element_wise:
                raise AssertionError(msg or str(e))

            # Provide detailed error information for element-wise comparison
            if array1.shape != array2.shape:
                error_msg = f"Arrays have different shapes: {array1.shape} vs {array2.shape}"
                self.fail(msg or error_msg)

            # Find differing elements
            diff_indices = np.where(
                ~np.isclose(array1, array2, rtol=10 ** (-decimal), atol=10 ** (-decimal))
            )

            if len(diff_indices[0]) > 0:
                error_msg = f"Arrays differ in {len(diff_indices[0])} elements\n"
                # Show first 10 differences
                max_diffs = min(10, len(diff_indices[0]))
                error_msg += "First differences:\n"

                for i in range(max_diffs):
                    index = tuple(diff_indices[j][i] for j in range(len(diff_indices)))
                    error_msg += f"  Index {index}: {array1[index]} vs {array2[index]}\n"

                self.fail(msg or error_msg)

    def create_mock_object(self, **attributes: Any) -> mock.MagicMock:
        """
        Create a mock object with the specified attributes.

        Args:
            **attributes: Attributes to set on the mock

        Returns:
            Configured MagicMock object
        """
        mock_obj = mock.MagicMock()
        for name, value in attributes.items():
            setattr(mock_obj, name, value)
        return mock_obj

    @staticmethod
    def parameterized_test(params: List[Tuple[Any, ...]]) -> Callable[[CallableT], CallableT]:
        """
        Decorator to parameterize a test method.

        This is a simple implementation for when pytest is not available.

        Example:
            @parameterized_test([
                ("case1", 1, 2, 3),
                ("case2", 4, 5, 9)
            ])
            def test_addition(self, name, a, b, expected):
                self.assertEqual(a + b, expected)

        Args:
            params: List of parameter tuples

        Returns:
            Decorated test function
        """

        def decorator(func: CallableT) -> CallableT:
            @wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
                for i, param_set in enumerate(params):
                    try:
                        func(self, *param_set, *args, **kwargs)
                    except Exception as e:
                        msg = f"Failed for params[{i}]: {param_set}"
                        raise type(e)(f"{msg}: {str(e)}") from e

            return cast(CallableT, wrapper)

        return decorator

    @contextmanager
    def mock_env_vars(self, **env_vars: str) -> Generator[None, None, None]:
        """
        Temporarily set environment variables for a test.

        Args:
            **env_vars: Environment variables to set

        Yields:
            None
        """
        original_values = {}

        # Save original values and set new values
        for key, value in env_vars.items():
            if key in os.environ:
                original_values[key] = os.environ[key]
            else:
                original_values[key] = None
            os.environ[key] = value

        try:
            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                if value is None:
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = value

    @contextmanager
    def mock_config(self, **config_values: Any) -> Generator[None, None, None]:
        """
        Temporarily override config values.

        Args:
            **config_values: Config values to set

        Yields:
            None
        """
        config_manager = get_config_manager()
        original_values = {}

        # Save original values and set new values
        for key, value in config_values.items():
            if key in config_manager._config_values:
                original_values[key] = config_manager._config_values[key]
            else:
                original_values[key] = None
            config_manager._config_values[key] = value

        try:
            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                if value is None:
                    if key in config_manager._config_values:
                        del config_manager._config_values[key]
                else:
                    config_manager._config_values[key] = value


class PerformanceTestCase(WDBXTestCase):
    """
    Test case for performance testing.

    Extends WDBXTestCase with additional utilities for benchmarking and performance testing.
    """

    # Default benchmark settings
    benchmark_results: Dict[str, Dict[str, Any]] = {}

    def setUp(self) -> None:
        """Set up performance test."""
        super().setUp()
        self.benchmark_results = {}

    def benchmark(
        self,
        func: Callable[..., Any],
        *args: Any,
        repeat: int = 3,
        name: Optional[str] = None,
        time_limit: Optional[float] = None,
        memory_limit_mb: Optional[float] = None,
        warmup: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Benchmark a function for performance.

        Args:
            func: Function to benchmark
            *args: Arguments to pass to the function
            repeat: Number of times to repeat the benchmark
            name: Name for this benchmark
            time_limit: Optional time limit in seconds
            memory_limit_mb: Optional memory limit in MB
            warmup: Number of warmup runs (not counted in results)
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary with benchmark results
        """
        if name is None:
            name = func.__name__

        logger.info(f"Starting benchmark '{name}' ({repeat} iterations)")

        # Perform warmup runs
        if warmup > 0:
            logger.debug(f"Performing {warmup} warmup runs...")
            for _ in range(warmup):
                func(*args, **kwargs)

        # Force garbage collection before starting
        gc.collect()

        # Record initial memory
        initial_memory = get_memory_usage()

        # Run benchmark
        times = []
        memory_readings = []

        for i in range(repeat):
            # Measure time
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()

            # Record time
            elapsed = end_time - start_time
            times.append(elapsed)

            # Measure memory
            gc.collect()  # Force GC to get accurate memory usage
            memory = get_memory_usage()
            memory_readings.append(memory)

            logger.debug(f"Run {i+1}/{repeat}: {elapsed:.6f}s")

        # Calculate statistics
        times = np.array(times)
        mean_time = float(np.mean(times))
        min_time = float(np.min(times))
        max_time = float(np.max(times))
        std_time = float(np.std(times))

        # Memory statistics (RSS in bytes)
        memory_usage = np.array([m["process"]["rss"] for m in memory_readings])
        mean_memory = float(np.mean(memory_usage))
        max_memory = float(np.max(memory_usage))

        # Memory change from start (in MB)
        memory_change_mb = (max_memory - initial_memory["process"]["rss"]) / (1024 * 1024)

        # Prepare result dictionary
        result = {
            "name": name,
            "iterations": repeat,
            "times": {
                "mean": mean_time,
                "min": min_time,
                "max": max_time,
                "std": std_time,
                "total": float(np.sum(times)),
                "raw": times.tolist(),
            },
            "memory": {
                "mean_bytes": mean_memory,
                "max_bytes": max_memory,
                "change_mb": memory_change_mb,
            },
            "timestamp": time.time(),
        }

        # Check against limits
        if time_limit is not None:
            self.assertLessEqual(
                mean_time,
                time_limit,
                f"Mean execution time {mean_time:.6f}s exceeds limit of {time_limit}s",
            )

        if memory_limit_mb is not None:
            self.assertLessEqual(
                memory_change_mb,
                memory_limit_mb,
                f"Memory usage {memory_change_mb:.2f}MB exceeds limit of {memory_limit_mb}MB",
            )

        # Store result
        self.benchmark_results[name] = result

        logger.info(
            f"Benchmark '{name}' completed: "
            f"{mean_time:.6f}s (±{std_time:.6f}s), "
            f"memory change: {memory_change_mb:.2f}MB"
        )

        return result

    def compare_benchmarks(
        self,
        baseline_name: str,
        new_name: str,
        time_ratio_limit: Optional[float] = None,
        memory_ratio_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compare two benchmarks.

        Args:
            baseline_name: Name of the baseline benchmark
            new_name: Name of the new benchmark
            time_ratio_limit: Optional limit for time ratio (new/baseline)
            memory_ratio_limit: Optional limit for memory ratio (new/baseline)

        Returns:
            Dictionary with comparison results
        """
        if baseline_name not in self.benchmark_results:
            self.fail(f"Baseline benchmark '{baseline_name}' not found")
        if new_name not in self.benchmark_results:
            self.fail(f"New benchmark '{new_name}' not found")

        baseline = self.benchmark_results[baseline_name]
        new = self.benchmark_results[new_name]

        # Calculate ratios
        time_ratio = new["times"]["mean"] / baseline["times"]["mean"]
        memory_ratio = (
            new["memory"]["change_mb"] / baseline["memory"]["change_mb"]
            if baseline["memory"]["change_mb"] != 0
            else float("inf") if new["memory"]["change_mb"] > 0 else 0
        )

        # Prepare comparison result
        result = {
            "baseline": baseline_name,
            "new": new_name,
            "time": {
                "baseline": baseline["times"]["mean"],
                "new": new["times"]["mean"],
                "ratio": time_ratio,
                "change_percent": (time_ratio - 1) * 100,
            },
            "memory": {
                "baseline": baseline["memory"]["change_mb"],
                "new": new["memory"]["change_mb"],
                "ratio": memory_ratio,
                "change_percent": (
                    (memory_ratio - 1) * 100 if memory_ratio != float("inf") else float("inf")
                ),
            },
        }

        # Log comparison
        mem_change_display = (
            f"{result['memory']['change_percent']:+.2f}%"
            if result["memory"]["change_percent"] != float("inf")
            else "N/A"
        )
        logger.info(
            f"Benchmark comparison: '{new_name}' vs '{baseline_name}'\n"
            f"  Time: {result['time']['new']:.6f}s vs {result['time']['baseline']:.6f}s "
            f"({result['time']['change_percent']:+.2f}%)\n"
            f"  Memory: {result['memory']['new']:.2f}MB vs {result['memory']['baseline']:.2f}MB "
            f"({mem_change_display})"
        )

        # Check limits
        if time_ratio_limit is not None:
            self.assertLessEqual(
                time_ratio,
                time_ratio_limit,
                f"Time ratio {time_ratio:.2f} exceeds limit of {time_ratio_limit}",
            )

        if memory_ratio_limit is not None and memory_ratio != float("inf"):
            self.assertLessEqual(
                memory_ratio,
                memory_ratio_limit,
                f"Memory ratio {memory_ratio:.2f} exceeds limit of {memory_ratio_limit}",
            )

        return result


class FixtureMixin:
    """
    Mixin providing pytest-like fixtures without requiring pytest.

    This provides a simple fixture implementation for unittest tests.
    """

    _fixtures: Dict[str, Any] = {}

    @classmethod
    def fixture(cls, func: Callable[[], T]) -> Callable[[], T]:
        """
        Decorator to create a fixture.

        Example:

        @FixtureMixin.fixture
        def sample_data():
            return {"name": "test", "value": 42}

        Args:
            func: Function that creates the fixture

        Returns:
            Fixture function
        """

        @wraps(func)
        def wrapper(self: Any) -> T:
            # Create a unique key for each test instance and fixture
            key = (id(self), func.__name__)

            # Check if we already have the fixture value
            if key not in cls._fixtures:
                cls._fixtures[key] = func(self)

            return cls._fixtures[key]

        return wrapper

    def tearDown(self) -> None:
        """Clean up fixtures"""
        # Clear fixtures for this test instance
        keys_to_remove = []
        for key in self._fixtures:
            if key[0] == id(self):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            value = self._fixtures.pop(key)

            # Call close() if available (for cleanup)
            if hasattr(value, "close") and callable(value.close):
                try:
                    value.close()
                except Exception:
                    pass

        super().tearDown()


class AsyncWDBXTestCase(WDBXTestCase):
    """
    Test case with support for asyncio coroutines and asynchronous testing.

    This test case provides utilities for testing async code.
    """

    _event_loop: Optional[asyncio.AbstractEventLoop] = None

    @contextmanager
    def event_loop(self) -> Generator[asyncio.AbstractEventLoop, None, None]:
        """
        Get or create an event loop for the test.

        This context manager ensures a clean event loop for each test.

        Yields:
            The event loop
        """
        # Store the original loop if it exists
        original_loop = asyncio.get_event_loop_policy().get_event_loop()

        # Create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._event_loop = loop

        try:
            yield loop
        finally:
            # Close the loop
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                if not loop.is_closed():
                    loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")

            # Restore the original loop
            asyncio.set_event_loop(original_loop)
            self._event_loop = None

    def run_async(self, coro: Any) -> Any:
        """
        Run a coroutine in the event loop.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        if self._event_loop is None:
            with self.event_loop() as loop:
                return loop.run_until_complete(coro)
        else:
            return self._event_loop.run_until_complete(coro)

    @contextmanager
    def assert_raises_async(
        self, expected_exception: Type[Exception]
    ) -> Generator[None, None, None]:
        """
        Assert that an async operation raises an exception.

        Args:
            expected_exception: The expected exception type

        Yields:
            None
        """
        raised = False
        actual_exception = None

        try:
            yield
        except Exception as e:
            raised = True
            actual_exception = e

        if not raised:
            self.fail(f"No exception was raised, expected {expected_exception.__name__}")

        if not isinstance(actual_exception, expected_exception):
            self.fail(
                f"Expected {expected_exception.__name__}, got {type(actual_exception).__name__}: {actual_exception}"
            )

    @contextmanager
    def measure_async_time(
        self, threshold: Optional[float] = None
    ) -> Generator[Dict[str, float], None, None]:
        """
        Measure execution time of an async block of code.

        Args:
            threshold: Optional threshold in seconds for test to pass

        Yields:
            Dictionary with timing information (start, end, duration)
        """
        result = {"start": time.time(), "end": 0, "duration": 0}

        try:
            yield result
        finally:
            result["end"] = time.time()
            result["duration"] = result["end"] - result["start"]

            logger.info(f"Async execution time: {result['duration']:.6f} seconds")

            if threshold is not None:
                self.assertLessEqual(
                    result["duration"],
                    threshold,
                    f"Async execution took {result['duration']:.6f}s, exceeding threshold of {threshold}s",
                )

    async def wait_for(
        self, condition: Callable[[], Any], timeout: float = 5.0, interval: float = 0.1
    ) -> None:
        """
        Wait asynchronously for a condition to become true.

        Args:
            condition: Function that returns a truthy value when condition is met
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds

        Raises:
            TimeoutError: If the condition is not met within the timeout
        """
        start_time = time.time()

        while True:
            if condition():
                return

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Condition not met within {timeout}s timeout")

            await asyncio.sleep(interval)


# Decorator utilities
def skip_if_missing_dependency(dependency: str) -> Callable[[CallableT], CallableT]:
    """
    Skip a test if a dependency is missing.

    Args:
        dependency: Module name to check for

    Returns:
        Decorator function
    """

    def decorator(test_func: CallableT) -> CallableT:
        @wraps(test_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                __import__(dependency)
                return test_func(*args, **kwargs)
            except ImportError:
                if isinstance(args[0], unittest.TestCase):
                    args[0].skipTest(f"Missing dependency: {dependency}")
                return None

        return cast(CallableT, wrapper)

    return decorator


def skip_on_platform(platform_name: str) -> Callable[[CallableT], CallableT]:
    """
    Skip a test on the specified platform.

    Args:
        platform_name: Platform to skip on (e.g., 'windows', 'linux', 'darwin')

    Returns:
        Decorator function
    """

    def decorator(test_func: CallableT) -> CallableT:
        @wraps(test_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_platform = platform.system().lower()
            if current_platform == platform_name.lower():
                if isinstance(args[0], unittest.TestCase):
                    args[0].skipTest(f"Test skipped on {platform_name}")
                return None
            return test_func(*args, **kwargs)

        return cast(CallableT, wrapper)

    return decorator


def skip_if_condition(
    condition: Callable[[], bool], reason: str
) -> Callable[[CallableT], CallableT]:
    """
    Skip a test if a condition is true.

    Args:
        condition: Function that returns True if the test should be skipped
        reason: Reason for skipping

    Returns:
        Decorator function
    """

    def decorator(test_func: CallableT) -> CallableT:
        @wraps(test_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if condition():
                if isinstance(args[0], unittest.TestCase):
                    args[0].skipTest(reason)
                return None
            return test_func(*args, **kwargs)

        return cast(CallableT, wrapper)

    return decorator


def retry_test(attempts: int = 3, delay: float = 0.5) -> Callable[[CallableT], CallableT]:
    """
    Retry a test if it fails.

    Args:
        attempts: Number of attempts to make
        delay: Delay between attempts in seconds

    Returns:
        Decorator function
    """

    def decorator(test_func: CallableT) -> CallableT:
        @wraps(test_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(attempts):
                try:
                    return test_func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < attempts - 1:
                        logger.info(f"Test failed, retrying ({attempt+1}/{attempts}): {str(e)}")
                        time.sleep(delay)

            if last_exception:
                raise last_exception

        return cast(CallableT, wrapper)

    return decorator


def run_tests(
    test_dir: Optional[Union[str, Path]] = None,
    pattern: str = "test_*.py",
    exclude_pattern: str = "skip_*.py",
    verbosity: int = 1,
) -> unittest.TestResult:
    """
    Run all tests in a directory.

    Args:
        test_dir: Directory containing tests (defaults to current directory)
        pattern: Pattern to match test files
        exclude_pattern: Pattern to exclude test files
        verbosity: Verbosity level for test output

    Returns:
        Test result object
    """
    if test_dir is None:
        test_dir = os.path.dirname(os.path.abspath(__file__))

    test_dir = Path(test_dir)

    # Dynamically discover tests
    loader = unittest.TestLoader()

    # Collect all test modules
    test_files = [f for f in test_dir.glob(pattern) if exclude_pattern not in f.name]
    suite = unittest.TestSuite()

    for test_file in test_files:
        module_name = test_file.stem
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if HAS_PYTEST:

    def pytest_fixture_to_unittest(fixture_func: Callable[..., T]) -> Callable[..., T]:
        """
        Convert a pytest fixture to a unittest compatible fixture.

        Args:
            fixture_func: Pytest fixture function

        Returns:
            Unittest compatible fixture function
        """

        @wraps(fixture_func)
        def wrapper(self: Any) -> T:
            # Get the fixture value using pytest's fixture mechanism

            # Create a dummy request object
            request = type("DummyRequest", (object,), {})()
            request.fixturenames = [fixture_func.__name__]
            request._fixturedef = None

            # Call the fixture function with the request
            return fixture_func(request)

        return wrapper


# If not using pytest, create a simple fixture implementation
if not HAS_PYTEST:

    def fixture(scope: str = "function") -> Callable[[CallableT], CallableT]:
        """
        Simple fixture decorator for compatibility with pytest.

        Args:
            scope: Fixture scope (function, class, module, session)

        Returns:
            Decorator function
        """

        def decorator(func: CallableT) -> CallableT:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            # Mark as a fixture
            wrapper._is_fixture = True
            wrapper._fixture_scope = scope
            return cast(CallableT, wrapper)

        return decorator


if __name__ == "__main__":
    # If this module is run directly, run all tests
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
