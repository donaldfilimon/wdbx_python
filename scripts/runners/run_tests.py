#!/usr/bin/env python
"""
Test runner for WDBX

This script simplifies running specific test suites or individual tests.
It automatically sets up the Python path and provides helpful commands.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("run_tests")

# Ensure the WDBX package is in the Python path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

def run_tests(
    test_paths: List[str], 
    verbose: bool = False, 
    coverage: bool = False,
    failfast: bool = False,
    junit_xml: bool = False,
    extra_args: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    skip_markers: Optional[List[str]] = None,
    test_pattern: Optional[str] = None,
) -> int:
    """
    Run pytest with the specified test paths and options.
    
    Args:
        test_paths: List of test paths to run
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        failfast: Stop on first failure
        junit_xml: Generate JUnit XML report
        extra_args: Additional pytest arguments
        markers: Run only tests with these markers
        skip_markers: Skip tests with these markers
        test_pattern: Pattern to filter test names
        
    Returns:
        Exit code from pytest
    """
    # Build pytest command
    cmd = ["pytest"]
    
    # Add flags
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=wdbx", "--cov-report=term-missing"])
    if failfast:
        cmd.append("-x")
    if junit_xml:
        cmd.append("--junitxml=test-results.xml")
        
    # Add marker filters
    if markers:
        marker_expr = " or ".join(markers)
        cmd.append(f"-m {marker_expr}")
    
    if skip_markers:
        skip_expr = " and ".join([f"not {m}" for m in skip_markers])
        if markers:
            cmd[-1] = f"{cmd[-1]} and {skip_expr}"
        else:
            cmd.append(f"-m {skip_expr}")
    
    # Add test name pattern
    if test_pattern:
        cmd.append(f"-k {test_pattern}")
        
    # Add extra args
    if extra_args:
        cmd.extend(extra_args)
        
    # Add test paths
    cmd.extend(test_paths)
    
    # Print command
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Track execution time
    start_time = time.time()
    result = subprocess.call(cmd)
    elapsed_time = time.time() - start_time
    
    # Log summary
    status = "PASSED" if result == 0 else "FAILED"
    logger.info(f"Tests {status} in {elapsed_time:.2f} seconds")
    
    return result

def find_test_files(pattern: str) -> List[str]:
    """
    Find test files matching the given pattern.
    
    Args:
        pattern: String pattern to match against file names
        
    Returns:
        List of matching file paths
    """
    matches = []
    
    # Handle full paths
    if os.path.isfile(pattern):
        return [pattern]
        
    # Handle pattern matching
    tests_dir = os.path.join(REPO_ROOT, "tests")
    
    # Check in all test directories
    test_dirs = ["unit", "integration", "test_plugins", "functional", "performance"]
    for dirname in test_dirs:
        dir_path = os.path.join(tests_dir, dirname)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.startswith("test_") and file.endswith(".py") and pattern.lower() in file.lower():
                    matches.append(os.path.join(dir_path, file))
    
    # Also check the tests directory directly
    if os.path.isdir(tests_dir):
        for file in os.listdir(tests_dir):
            if file.startswith("test_") and file.endswith(".py") and pattern.lower() in file.lower():
                matches.append(os.path.join(tests_dir, file))
    
    return matches

def discover_test_markers() -> Dict[str, Set[str]]:
    """
    Discover available test markers in the project.
    
    Returns:
        Dictionary mapping test files to sets of markers they contain
    """
    markers: Dict[str, Set[str]] = {}
    tests_dir = os.path.join(REPO_ROOT, "tests")
    
    # Check for common markers in all Python test files
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, REPO_ROOT)
                
                # Extract markers from the file
                file_markers = set()
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                        
                    # Check for @pytest.mark.X decorators
                    import re
                    marker_pattern = r"@pytest\.mark\.(\w+)"
                    for match in re.finditer(marker_pattern, content):
                        file_markers.add(match.group(1))
                        
                    if file_markers:
                        markers[rel_path] = file_markers
                except (OSError, UnicodeDecodeError):
                    pass
    
    return markers

def list_available_tests() -> None:
    """List all available test files and markers in the repository."""
    tests_dir = os.path.join(REPO_ROOT, "tests")
    
    if not os.path.isdir(tests_dir):
        logger.error(f"No tests directory found at {tests_dir}")
        return
    
    print("\n=== Available Test Categories ===\n")
    
    # List all test directories and files
    test_dirs = {
        "Unit Tests": os.path.join(tests_dir, "unit"),
        "Integration Tests": os.path.join(tests_dir, "integration"),
        "Plugin Tests": os.path.join(tests_dir, "test_plugins"),
        "Functional Tests": os.path.join(tests_dir, "functional"),
        "Performance Tests": os.path.join(tests_dir, "performance"),
    }
    
    for category, dir_path in test_dirs.items():
        if os.path.isdir(dir_path):
            test_files = [f for f in os.listdir(dir_path) if f.startswith("test_") and f.endswith(".py")]
            if test_files:
                print(f"\n{category}:")
                for test in sorted(test_files):
                    print(f"  • {test[:-3]}")  # Remove .py extension
    
    # List tests in the root of the tests directory
    if os.path.isdir(tests_dir):
        root_tests = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]
        if root_tests:
            print("\nMiscellaneous Tests:")
            for test in sorted(root_tests):
                print(f"  • {test[:-3]}")  # Remove .py extension
    
    # Find all markers
    markers = discover_test_markers()
    if markers:
        all_markers = set()
        for marker_set in markers.values():
            all_markers.update(marker_set)
        
        if all_markers:
            print("\n=== Available Test Markers ===\n")
            for marker in sorted(all_markers):
                print(f"  • {marker}")
            
            print("\nExample usage:")
            print("  Run tests with specific marker:")
            print("    python run_tests.py --markers integration")
            print("  Run tests excluding specific marker:")
            print("    python run_tests.py --skip-markers slow")
    
    # Examples
    print("\n=== Example Commands ===\n")
    print("  Run all tests:")
    print("    python run_tests.py")
    print("  Run with coverage:")
    print("    python run_tests.py --coverage")
    print("  Run specific test category:")
    print("    python run_tests.py --unit")
    print("  Run specific test file:")
    print("    python run_tests.py tests/unit/test_cache.py")
    print("  Run tests matching a pattern:")
    print("    python run_tests.py --pattern 'cache'")
    print("  Run tests with specific markers:")
    print("    python run_tests.py --markers 'integration,plugin'")
    print("  Skip slow tests:")
    print("    python run_tests.py --skip-markers 'slow'")

def main() -> int:
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run WDBX tests")
    
    # Test filtering options
    test_group = parser.add_argument_group("Test Selection")
    test_group.add_argument("tests", nargs="*", help="Test files or patterns to run")
    test_group.add_argument("-p", "--pattern", help="Filter tests by pattern")
    test_group.add_argument("-m", "--markers", help="Run tests with specific markers (comma-separated)")
    test_group.add_argument("-M", "--skip-markers", help="Skip tests with specific markers (comma-separated)")
    
    # Test categories
    categories = parser.add_argument_group("Test Categories")
    categories.add_argument("--unit", action="store_true", help="Run all unit tests")
    categories.add_argument("--integration", action="store_true", help="Run all integration tests")
    categories.add_argument("--plugin", action="store_true", help="Run all plugin tests")
    categories.add_argument("--functional", action="store_true", help="Run all functional tests")
    categories.add_argument("--performance", action="store_true", help="Run all performance tests")
    
    # Test options
    options = parser.add_argument_group("Test Options")
    options.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    options.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    options.add_argument("-f", "--failfast", action="store_true", help="Stop on first failure")
    options.add_argument("-j", "--junit-xml", action="store_true", help="Generate JUnit XML report")
    options.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
    options.add_argument("-l", "--list", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # List tests if requested
    if args.list:
        list_available_tests()
        return 0
    
    # Collect test paths
    test_paths = []
    
    # Check if specific test categories are requested
    if args.unit:
        test_paths.append("tests/unit/")
    if args.integration:
        test_paths.append("tests/integration/")
    if args.plugin:
        test_paths.append("tests/test_plugins/")
    if args.functional:
        test_paths.append("tests/functional/")
    if args.performance:
        test_paths.append("tests/performance/")
        
    # If no specific categories and no specific tests, run all tests
    if not test_paths and not args.tests:
        test_paths.append("tests/")
    
    # Process test patterns
    for pattern in args.tests:
        matching_files = find_test_files(pattern)
        if matching_files:
            test_paths.extend(matching_files)
        else:
            # If it doesn't match any files, try passing it directly to pytest
            # This lets users specify test names like "test_cache_hit"
            test_paths.append(pattern)
    
    # Split comma-separated markers
    markers = args.markers.split(",") if args.markers else None
    skip_markers = args.skip_markers.split(",") if args.skip_markers else None
    
    # Run the tests
    return run_tests(
        test_paths=test_paths,
        verbose=args.verbose,
        coverage=args.coverage,
        failfast=args.failfast,
        junit_xml=args.junit_xml,
        test_pattern=args.pattern,
        markers=markers,
        skip_markers=skip_markers
    )

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error running tests: {e}")
        sys.exit(1) 