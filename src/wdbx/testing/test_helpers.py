"""
Test helpers for WDBX.

This module provides utilities for testing WDBX components, including test case mixins,
fixtures, and helper functions.
"""

import contextlib
import gc
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union, cast

import numpy as np

from ..config.config_manager import Environment, get_config_manager
from ..utils.logging_utils import LogFormat, get_logger, log_context
from ..utils.memory_utils import get_memory_usage

logger = get_logger("WDBX.Testing")


class WDBXTestCase(unittest.TestCase):
    """
    Base test case for WDBX tests.
    
    Provides common functionality for WDBX tests, including:
    - Temporary directory creation and cleanup
    - Configuration management
    - Log capture
    - Performance measurement
    """
    
    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level resources."""
        super().setUpClass()
        
        # Create a temporary directory for tests
        cls.temp_dir = tempfile.mkdtemp(prefix="wdbx_test_")
        
        # Configure the test environment
        os.environ["WDBX_ENV"] = "testing"
        os.environ["WDBX_DATA_DIR"] = os.path.join(cls.temp_dir, "data")
        os.environ["WDBX_LOG_DIR"] = os.path.join(cls.temp_dir, "logs")
        os.environ["WDBX_CONFIG_DIR"] = os.path.join(cls.temp_dir, "config")
        os.environ["WDBX_LOG_TO_FILE"] = "false"
        os.environ["WDBX_MEMORY_OPTIMIZATION_ENABLED"] = "false"
        
        # Create standard directories
        for dir_name in ["data", "logs", "config"]:
            os.makedirs(os.path.join(cls.temp_dir, dir_name), exist_ok=True)
        
        # Reset config manager to pick up new environment
        config_manager = get_config_manager()
        config_manager._config_values = {}
        config_manager._loaded_sources = set()
        config_manager.load_all()
        
        logger.info(f"Test class {cls.__name__} set up, temp dir: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up class-level resources."""
        # Remove temporary directory
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            logger.info(f"Removed temporary directory {cls.temp_dir}")
        
        # Restore environment variables
        for var in ["WDBX_ENV", "WDBX_DATA_DIR", "WDBX_LOG_DIR", "WDBX_CONFIG_DIR",
                   "WDBX_LOG_TO_FILE", "WDBX_MEMORY_OPTIMIZATION_ENABLED"]:
            if var in os.environ:
                del os.environ[var]
        
        # Reset config manager
        config_manager = get_config_manager()
        config_manager._config_values = {}
        config_manager._loaded_sources = set()
        
        # Run garbage collection to clean up resources
        gc.collect()
        
        super().tearDownClass()
    
    def setUp(self) -> None:
        """Set up test-level resources."""
        super().setUp()
        
        # Create a test-specific temporary directory
        self.test_temp_dir = os.path.join(self.temp_dir, self.id().split(".")[-1])
        os.makedirs(self.test_temp_dir, exist_ok=True)
        
        # Set up log context for the test
        self._log_context = log_context(
            component="Test",
            tags=["test", self.__class__.__name__],
            test_name=self.id()
        )
        self._log_context.__enter__()
        
        logger.info(f"Starting test {self.id()}")
    
    def tearDown(self) -> None:
        """Clean up test-level resources."""
        logger.info(f"Finished test {self.id()}")
        
        # Exit log context
        self._log_context.__exit__(None, None, None)
        
        # Clean up test-specific temp directory
        if hasattr(self, "test_temp_dir") and os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)
        
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
        root_logger.addHandler(handler)
        
        try:
            yield log_capture
        finally:
            # Remove the handler
            root_logger.removeHandler(handler)
    
    @contextmanager
    def assert_logs(self, level: str = "INFO", message: Optional[str] = None) -> Generator[None, None, None]:
        """
        Assert that logs are generated during a block.
        
        Args:
            level: Minimum log level to capture
            message: Optional message that should appear in the logs
            
        Yields:
            None
        """
        with self.capture_logs(level) as logs:
            yield
            
            log_content = logs.getvalue()
            self.assertTrue(log_content, f"No logs were generated at level {level} or higher")
            
            if message:
                self.assertIn(message, log_content, f"Expected message not found in logs: {message}")
    
    @contextmanager
    def measure_time(self, threshold: Optional[float] = None) -> Generator[Dict[str, float], None, None]:
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
                    f"Execution took {result['duration']:.6f}s, exceeding threshold of {threshold}s"
                )
    
    @contextmanager
    def measure_memory(self, threshold_mb: Optional[float] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Measure memory usage of a block of code.
        
        Args:
            threshold_mb: Optional threshold in MB for test to pass
            
        Yields:
            Dictionary with memory usage information
        """
        before = get_memory_usage()
        result = {"before": before, "after": {}, "diff": {}}
        
        try:
            yield result
        finally:
            # Force garbage collection
            gc.collect()
            
            # Get memory usage after execution
            after = get_memory_usage()
            result["after"] = after
            
            # Calculate difference
            diff_rss = after["process"]["rss"] - before["process"]["rss"]
            diff_percent = after["process"]["percent"] - before["process"]["percent"]
            
            result["diff"] = {
                "rss": diff_rss,
                "percent": diff_percent,
                "mb": diff_rss / (1024 * 1024)
            }
            
            logger.info(f"Memory change: {result['diff']['mb']:.2f} MB")
            
            if threshold_mb is not None:
                self.assertLessEqual(
                    result["diff"]["mb"],
                    threshold_mb,
                    f"Memory usage increased by {result['diff']['mb']:.2f} MB, "
                    f"exceeding threshold of {threshold_mb} MB"
                )
    
    def create_temp_file(self, content: str, name: Optional[str] = None) -> str:
        """
        Create a temporary file with the specified content.
        
        Args:
            content: Content to write to the file
            name: Optional file name (default: random name)
            
        Returns:
            Path to the created file
        """
        if name is None:
            handle, path = tempfile.mkstemp(dir=self.test_temp_dir)
            os.close(handle)
        else:
            path = os.path.join(self.test_temp_dir, name)
        
        with open(path, "w") as f:
            f.write(content)
        
        return path
    
    def create_temp_dir(self, name: Optional[str] = None) -> str:
        """
        Create a temporary directory.
        
        Args:
            name: Optional directory name (default: random name)
            
        Returns:
            Path to the created directory
        """
        if name is None:
            path = tempfile.mkdtemp(dir=self.test_temp_dir)
        else:
            path = os.path.join(self.test_temp_dir, name)
            os.makedirs(path, exist_ok=True)
        
        return path
    
    def assert_files_equal(self, file1: Union[str, Path], file2: Union[str, Path]) -> None:
        """
        Assert that two files have the same content.
        
        Args:
            file1: Path to first file
            file2: Path to second file
        """
        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            content1 = f1.read()
            content2 = f2.read()
            
            self.assertEqual(
                content1, content2,
                f"Files {file1} and {file2} have different content"
            )
    
    def assert_json_equal(self, json1: Any, json2: Any, msg: Optional[str] = None) -> None:
        """
        Assert that two JSON objects are equal, with pretty error messages.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            msg: Optional message for assertion
        """
        import json
        
        # Helper function to format JSON for readability in error messages
        def format_json(obj: Any) -> str:
            return json.dumps(obj, indent=2, sort_keys=True)
        
        try:
            self.assertEqual(json1, json2, msg)
        except AssertionError as e:
            # Add formatted JSON to the error message
            error_msg = f"{e}\n\nFirst JSON:\n{format_json(json1)}\n\nSecond JSON:\n{format_json(json2)}"
            raise AssertionError(error_msg) from None
    
    def assert_dict_contains_subset(self, subset: Dict[Any, Any], dictionary: Dict[Any, Any], msg: Optional[str] = None) -> None:
        """
        Assert that dictionary contains all key-value pairs from subset.
        
        Args:
            subset: Dictionary with key-value pairs that should be in dictionary
            dictionary: Dictionary to check
            msg: Optional message for assertion
        """
        missing = {}
        mismatched = {}
        
        for key, value in subset.items():
            if key not in dictionary:
                missing[key] = value
            elif value != dictionary[key]:
                mismatched[key] = (value, dictionary[key])
        
        if missing or mismatched:
            standard_msg = ""
            if missing:
                standard_msg += f"Missing key-value pairs: {missing}\n"
            if mismatched:
                standard_msg += f"Mismatched key-value pairs: {mismatched}"
            
            self.fail(self._formatMessage(msg, standard_msg))
    
    def assert_arrays_equal(
        self, 
        array1: np.ndarray, 
        array2: np.ndarray, 
        decimal: int = 7,
        msg: Optional[str] = None
    ) -> None:
        """
        Assert that two numpy arrays are equal, within a tolerance.
        
        Args:
            array1: First array
            array2: Second array
            decimal: Number of decimal places to check
            msg: Optional message for assertion
        """
        try:
            np.testing.assert_almost_equal(array1, array2, decimal=decimal)
        except AssertionError as e:
            # Format error message
            if array1.size <= 10 and array2.size <= 10:
                error_msg = f"{e}\n\nArray1:\n{array1}\n\nArray2:\n{array2}"
            else:
                error_msg = f"{e}\n\nShape1: {array1.shape}, Shape2: {array2.shape}"
                
                # Add info about differences
                abs_diff = np.abs(array1 - array2)
                max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
                max_diff = abs_diff[max_diff_idx]
                
                error_msg += f"\nMax difference: {max_diff} at index {max_diff_idx}"
                error_msg += f"\nArray1 value: {array1[max_diff_idx]}, Array2 value: {array2[max_diff_idx]}"
                
                # Add basic stats
                error_msg += f"\nArray1 min/max/mean: {array1.min()}/{array1.max()}/{array1.mean()}"
                error_msg += f"\nArray2 min/max/mean: {array2.min()}/{array2.max()}/{array2.mean()}"
            
            if msg:
                error_msg = f"{msg}: {error_msg}"
                
            raise AssertionError(error_msg) from None


class PerformanceTestCase(WDBXTestCase):
    """
    Test case for performance testing.
    
    Extends WDBXTestCase with additional utilities for benchmarking and performance testing.
    """
    
    def setUp(self) -> None:
        """Set up test-level resources."""
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
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Benchmark a function for time and memory usage.
        
        Args:
            func: Function to benchmark
            *args: Arguments for the function
            repeat: Number of times to repeat the benchmark
            name: Name for the benchmark (default: function name)
            time_limit: Optional time limit in seconds
            memory_limit_mb: Optional memory limit in MB
            warmup: Number of warmup iterations
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dictionary with benchmark results
        """
        if name is None:
            name = func.__name__
        
        logger.info(f"Benchmarking {name} ({repeat} iterations)...")
        
        # Warmup
        if warmup > 0:
            logger.debug(f"Warming up {name} ({warmup} iterations)...")
            for _ in range(warmup):
                func(*args, **kwargs)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for i in range(repeat):
            # Force garbage collection before each iteration
            gc.collect()
            
            # Get initial memory usage
            before_memory = get_memory_usage()
            
            # Measure time
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Force garbage collection after the function call
            gc.collect()
            
            # Get final memory usage
            after_memory = get_memory_usage()
            memory_diff = (after_memory["process"]["rss"] - before_memory["process"]["rss"]) / (1024 * 1024)
            
            times.append(elapsed)
            memory_usage.append(memory_diff)
            
            logger.debug(f"Iteration {i+1}/{repeat}: {elapsed:.6f}s, {memory_diff:.2f} MB")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        avg_memory = sum(memory_usage) / len(memory_usage)
        min_memory = min(memory_usage)
        max_memory = max(memory_usage)
        
        logger.info(
            f"Benchmark {name} results:\n"
            f"  Time: avg={avg_time:.6f}s, min={min_time:.6f}s, max={max_time:.6f}s\n"
            f"  Memory: avg={avg_memory:.2f} MB, min={min_memory:.2f} MB, max={max_memory:.2f} MB"
        )
        
        # Check limits
        if time_limit is not None:
            self.assertLessEqual(
                avg_time, 
                time_limit, 
                f"Average time ({avg_time:.6f}s) exceeds limit ({time_limit}s)"
            )
        
        if memory_limit_mb is not None:
            self.assertLessEqual(
                avg_memory, 
                memory_limit_mb, 
                f"Average memory usage ({avg_memory:.2f} MB) exceeds limit ({memory_limit_mb} MB)"
            )
        
        # Store results
        benchmark_result = {
            "name": name,
            "iterations": repeat,
            "time": {
                "avg": avg_time,
                "min": min_time,
                "max": max_time,
                "values": times,
            },
            "memory": {
                "avg": avg_memory,
                "min": min_memory,
                "max": max_memory,
                "values": memory_usage,
            },
            "limits": {
                "time": time_limit,
                "memory": memory_limit_mb,
            },
            "result": result,
        }
        
        self.benchmark_results[name] = benchmark_result
        return benchmark_result
    
    def compare_benchmarks(
        self, 
        baseline_name: str, 
        new_name: str, 
        time_ratio_limit: Optional[float] = None,
        memory_ratio_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compare two benchmarks and assert performance ratios.
        
        Args:
            baseline_name: Name of the baseline benchmark
            new_name: Name of the new benchmark
            time_ratio_limit: Optional limit for time ratio (new/baseline)
            memory_ratio_limit: Optional limit for memory ratio (new/baseline)
            
        Returns:
            Dictionary with comparison results
        """
        # Get benchmark results
        baseline = self.benchmark_results.get(baseline_name)
        new = self.benchmark_results.get(new_name)
        
        if baseline is None:
            self.fail(f"Baseline benchmark {baseline_name} not found")
        
        if new is None:
            self.fail(f"New benchmark {new_name} not found")
        
        # Calculate ratios
        time_ratio = new["time"]["avg"] / baseline["time"]["avg"]
        memory_ratio = new["memory"]["avg"] / baseline["memory"]["avg"] if baseline["memory"]["avg"] != 0 else float('inf')
        
        logger.info(
            f"Benchmark comparison {new_name} vs {baseline_name}:\n"
            f"  Time: {new['time']['avg']:.6f}s vs {baseline['time']['avg']:.6f}s "
            f"(ratio: {time_ratio:.2f})\n"
            f"  Memory: {new['memory']['avg']:.2f} MB vs {baseline['memory']['avg']:.2f} MB "
            f"(ratio: {memory_ratio:.2f})"
        )
        
        # Check limits
        if time_ratio_limit is not None:
            self.assertLessEqual(
                time_ratio, 
                time_ratio_limit, 
                f"Time ratio ({time_ratio:.2f}) exceeds limit ({time_ratio_limit})"
            )
        
        if memory_ratio_limit is not None:
            self.assertLessEqual(
                memory_ratio, 
                memory_ratio_limit, 
                f"Memory ratio ({memory_ratio:.2f}) exceeds limit ({memory_ratio_limit})"
            )
        
        # Return comparison results
        return {
            "baseline": baseline_name,
            "new": new_name,
            "time_ratio": time_ratio,
            "memory_ratio": memory_ratio,
            "limits": {
                "time_ratio": time_ratio_limit,
                "memory_ratio": memory_ratio_limit,
            },
        }


class AsyncWDBXTestCase(WDBXTestCase):
    """
    Test case for asynchronous tests.
    
    Extends WDBXTestCase with utilities for testing asynchronous code.
    """
    
    @contextmanager
    def event_loop(self) -> Generator[Any, None, None]:
        """
        Create an event loop for testing asynchronous code.
        
        This implementation uses asyncio's default event loop.
        Subclasses can override to use a different event loop implementation.
        
        Yields:
            Event loop
        """
        import asyncio
        
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            yield loop
        finally:
            # Clean up the event loop
            loop.close()
            asyncio.set_event_loop(None)
    
    def run_async(self, coro: Any) -> Any:
        """
        Run an asynchronous coroutine in a new event loop.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        import asyncio
        
        with self.event_loop() as loop:
            return loop.run_until_complete(coro)
    
    @contextmanager
    def assert_raises_async(self, expected_exception: Type[Exception]) -> Generator[None, None, None]:
        """
        Assert that an asynchronous block raises an expected exception.
        
        This should be used with run_async() in a lambda function.
        
        Example:
            with self.assert_raises_async(ValueError):
                self.run_async(lambda: my_async_function())
        
        Args:
            expected_exception: Expected exception class
            
        Yields:
            None
        """
        try:
            yield
            self.fail(f"Expected {expected_exception.__name__} not raised")
        except expected_exception:
            pass  # This is expected
        except Exception as e:
            self.fail(f"Expected {expected_exception.__name__}, but got {type(e).__name__}: {e}")
    
    @contextmanager
    def measure_async_time(self, threshold: Optional[float] = None) -> Generator[Dict[str, float], None, None]:
        """
        Measure execution time of an asynchronous block of code.
        
        This should be used with run_async() in a lambda function.
        
        Example:
            with self.measure_async_time() as result:
                self.run_async(lambda: my_async_function())
                
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
                    f"Async execution took {result['duration']:.6f}s, exceeding threshold of {threshold}s"
                )


def skip_if_missing_dependency(dependency: str) -> Callable[[Callable], Callable]:
    """
    Decorator to skip a test if a dependency is missing.
    
    Args:
        dependency: Name of the dependency to check
        
    Returns:
        Decorator function
    """
    import importlib
    
    def decorator(test_func: Callable) -> Callable:
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(dependency)
            except ImportError:
                raise unittest.SkipTest(f"Dependency {dependency} not available")
            return test_func(*args, **kwargs)
        return wrapper
    
    return decorator


def skip_on_platform(platform: str) -> Callable[[Callable], Callable]:
    """
    Decorator to skip a test on a specific platform.
    
    Args:
        platform: Platform name (e.g., 'win32', 'linux', 'darwin')
        
    Returns:
        Decorator function
    """
    import platform as plt
    
    def decorator(test_func: Callable) -> Callable:
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            if plt.system().lower() == platform.lower() or plt.platform().startswith(platform.lower()):
                raise unittest.SkipTest(f"Test skipped on {platform}")
            return test_func(*args, **kwargs)
        return wrapper
    
    return decorator


def run_tests(test_dir: Optional[str] = None, pattern: str = 'test_*.py') -> unittest.TestResult:
    """
    Run WDBX tests.
    
    Args:
        test_dir: Directory containing test files (default: current directory)
        pattern: Pattern for test file names
        
    Returns:
        Test result object
    """
    if test_dir is None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up test environment
    os.environ["WDBX_ENV"] = "testing"
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_dir, pattern=pattern)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    return test_result


if __name__ == "__main__":
    # If this module is run directly, run all tests
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1) 