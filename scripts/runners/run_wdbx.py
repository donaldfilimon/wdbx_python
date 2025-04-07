#!/usr/bin/env python3
"""
Run WDBX directly from the source directory.

This script is a convenience utility for running WDBX commands without installing
the package. It searches for the WDBX package, adds it to the Python path, and
runs the specified command.
"""

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("wdbx_runner")

def get_platform_info() -> Dict[str, str]:
    """
    Get information about the current platform.
    
    Returns:
        Dictionary with system information
    """
    try:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "cpu_count": str(os.cpu_count() or "Unknown"),
            "node": platform.node()
        }
    except Exception as e:
        logger.warning(f"Error getting platform info: {e}")
        return {
            "system": "Unknown",
            "python_version": ".".join(map(str, sys.version_info[:3]))
        }

def check_python_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if essential Python dependencies are installed.
    
    Returns:
        Tuple of (all_dependencies_present, missing_dependencies)
    """
    essential_packages = ["numpy", "scikit-learn", "aiohttp"]
    missing = []
    
    for package in essential_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def print_status(message: str, error: bool = False, debug: bool = False) -> None:
    """
    Print a status message with timestamp.
    
    Args:
        message: Message to print
        error: Whether this is an error message
        debug: Whether this is a debug message
    """
    if debug:
        logger.debug(message)
    elif error:
        logger.error(message)
    else:
        logger.info(message)

def find_wdbx_dir() -> Optional[Path]:
    """
    Find the WDBX package directory.
    
    Returns:
        Path to the WDBX package directory, or None if not found
    """
    # Get script directory and project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    
    # Check if we're in the WDBX project directory
    wdbx_dir = project_root / "src" / "wdbx"
    if wdbx_dir.exists() and wdbx_dir.is_dir():
        return wdbx_dir
    
    # Check a few other places
    possible_paths = [
        project_root / "wdbx",
        script_dir.parent / "src" / "wdbx",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path
    
    # Try checking if wdbx is installed in site-packages
    try:
        import wdbx
        return Path(wdbx.__file__).parent
    except ImportError:
        pass
    
    return None

def check_data_directory(wdbx_dir: Path) -> bool:
    """
    Check if the WDBX data directory exists and is writable.
    
    Args:
        wdbx_dir: Path to the WDBX package directory
        
    Returns:
        True if the data directory exists and is writable
    """
    # Check several potential data directory locations
    data_dirs = [
        wdbx_dir.parent.parent / "wdbx_data",
        Path.home() / ".wdbx" / "data"
    ]
    
    for data_dir in data_dirs:
        if data_dir.exists():
            # Check if directory is writable
            try:
                test_file = data_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                logger.debug(f"Data directory found and writable: {data_dir}")
                return True
            except (OSError, PermissionError):
                logger.warning(f"Data directory found but not writable: {data_dir}")
    
    # No existing data directory found, try to create one
    try:
        default_data_dir = data_dirs[0]
        default_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new data directory: {default_data_dir}")
        return True
    except (OSError, PermissionError) as e:
        logger.error(f"Could not create data directory: {e}")
        return False

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run WDBX directly from the source directory",
        epilog="Example: python run_wdbx.py --interactive"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run WDBX in interactive mode"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to run the HTTP server on (default: 8080)"
    )
    
    parser.add_argument(
        "--socket-port", "-s",
        type=int,
        default=9090,
        help="Port to run the socket server on (default: 9090)"
    )
    
    parser.add_argument(
        "--no-socket",
        action="store_true",
        help="Do not start the socket server"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment and exit without running WDBX"
    )
    
    return parser.parse_args()

def main() -> int:
    """
    Main entry point to find and run WDBX.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Get platform info
    platform_info = get_platform_info()
    logger.info(f"Running on {platform_info['system']} {platform_info.get('release', '')} with Python {platform_info['python_version']}")
    
    # Check dependencies
    deps_ok, missing_deps = check_python_dependencies()
    if not deps_ok:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.warning("You might experience issues when running WDBX.")
        logger.warning("Consider installing them with: pip install " + " ".join(missing_deps))
    
    # Find WDBX directory
    wdbx_dir = find_wdbx_dir()

    if wdbx_dir is None:
        logger.error("Could not find the 'wdbx' package directory.")
        logger.error("Please run this script from the project's root directory")
        logger.error("or ensure 'wdbx' is installed correctly.")
        return 1

    logger.info(f"Found WDBX package at: {wdbx_dir}")
    
    # Check data directory
    if not check_data_directory(wdbx_dir):
        logger.warning("Could not find or create a writable data directory")
        logger.warning("WDBX might not be able to save data")
    
    # Add the directory *containing* the 'wdbx' package to the Python path
    package_parent_dir = os.path.dirname(wdbx_dir)
    if package_parent_dir not in sys.path:
        sys.path.insert(0, package_parent_dir)
        logger.debug(f"Added {package_parent_dir} to Python path")

    # Exit if only checking the environment
    if args.check:
        logger.info("Environment check complete")
        return 0

    # Convert command line arguments for better compatibility
    wdbx_args = []
    
    # Handle interactive mode
    if args.interactive:
        wdbx_args.append("--interactive")
    
    # Handle port settings
    if args.port != 8080:
        wdbx_args.extend(["--port", str(args.port)])
    
    # Handle socket port settings
    if args.no_socket:
        wdbx_args.append("--no-socket")
    elif args.socket_port != 9090:
        wdbx_args.extend(["--socket-port", str(args.socket_port)])
    
    # Add log level
    wdbx_args.extend(["--log-level", args.log_level])
    
    # Try to import from the entry points module
    try:
        logger.debug("Attempting to run via entry_points module...")
        from wdbx.ui.cli import cli_main
        logger.info("Running WDBX from entry_points module...")
        
        # Pass all command line arguments to the CLI module
        sys.argv[0] = "wdbx"  # Replace script name for better help messages
        exit_code = cli_main(wdbx_args)
        return exit_code
    except (ImportError, AttributeError) as e:
        logger.error(f"Could not import entry_points module: {e}")
        
        # Fallback to running as a module
        try:
            logger.debug("Attempting to run as a module...")
            import wdbx
            logger.info(f"Found WDBX module version: {getattr(wdbx, '__version__', 'unknown')}")
            
            # Run as a module
            cmd = [sys.executable, "-m", "wdbx"] + wdbx_args
            logger.debug(f"Running command: {' '.join(cmd)}")
            return subprocess.call(cmd)
        except (ImportError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to run WDBX: {e}")
            return 1
        
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("WDBX was interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)